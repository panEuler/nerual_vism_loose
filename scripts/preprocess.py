#!/usr/bin/env python
"""
preprocess.py
========================
从 PDB 文件中提取原子级特征:坐标、原子类型、半径、LJ 参数和电荷。

流程:
  1. reduce -Trim:  去除已有氢原子
  2. BioPython:     提取单链（仅标准氨基酸）
  3. PDBFixer:      修补缺失原子、替换非标准残基
  4. OpenMM Modeller: 重新加氢(AMBER 兼容命名)
  5. OpenMM + AMBER ff14SB: 力场参数化 → 提取 LJ 参数和电荷

用法:
  # 单个 PDB 文件:
  python process.py <pdb_file> <chain_id> [output_dir]

  # 批量处理文件夹中所有 .pdb 文件:
  python process.py --dir <pdb_dir> <chain_id> [output_dir]

示例:
  python process.py 1MBN.pdb A ./output/
  python process.py --dir ./raw_pdbs/ A ./output/

输出 (.npy):
  {prefix}_coords.npy       [N, 3]  原子坐标 (Å)
  {prefix}_atom_types.npy    [N]     元素类型 (H, C, N, O, S ...)
  {prefix}_atom_names.npy    [N]     力场原子名 (CA, CB, NZ ...)
  {prefix}_radii.npy         [N]     vdW 半径 σ/2 (Å)
  {prefix}_epsilon.npy       [N]     LJ well depth ε (kJ/mol)
  {prefix}_sigma.npy         [N]     LJ σ (Å)
  {prefix}_charges.npy       [N]     部分电荷 (e, 基本电荷单位)
  {prefix}_res_names.npy     [N]     残基名 (ALA, GLY ...)
  {prefix}_res_ids.npy       [N]     残基编号

依赖:
  pip install biopython openmm pdbfixer numpy
  reduce 程序需要在 PATH 中 (http://kinemage.biochem.duke.edu/software/reduce.php)
"""

import os
import sys
import tempfile
import numpy as np
from subprocess import Popen, PIPE


# ==============================================================================
# Step 1: reduce 去氢
# ==============================================================================
def remove_hydrogens_with_reduce(in_pdb, out_pdb):
    """
    使用 reduce -Trim 去除 PDB 中已有的氢原子。
    
    为什么要先去氢再重新加氢？
    - 原始 PDB 中的氢原子可能不全、位置不对、或命名不一致
    - 先统一去掉，再由 OpenMM 按 AMBER 力场规范重新添加
    """
    args = ["reduce", "-Trim", in_pdb]
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()

    # reduce 的正常输出也会包含一些 warning，returncode != 0 不一定是错误
    with open(out_pdb, "w") as f:
        f.write(stdout.decode("utf-8").rstrip())

    print(f"  [reduce] Removed existing hydrogens → {out_pdb}")


# ==============================================================================
# Step 2: BioPython 提取单链
# ==============================================================================
def extract_chain(in_pdb, out_pdb, chain_id):
    """
    使用 BioPython 从 PDB 中提取指定链。
    
    选择逻辑:
    - 只保留指定 chain_id 的链
    - 只保留标准氨基酸残基（排除 HETATM 中的水、配体等）
    - 排除无序原子（保留 altloc A 或 1）
    """
    from Bio.PDB import PDBParser, PDBIO, Select

    class ChainAndResSelect(Select):
        def __init__(self, target_chain):
            self.target_chain = target_chain

        def accept_chain(self, chain):
            return chain.get_id() == self.target_chain

        def accept_residue(self, residue):
            # het_flag == " " 表示标准氨基酸
            return residue.get_id()[0] == " "

        def accept_atom(self, atom):
            if atom.is_disordered():
                return atom.get_altloc() in ("A", "1")
            return True

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("protein", in_pdb)
    io = PDBIO()
    io.set_structure(struct)
    io.save(out_pdb, select=ChainAndResSelect(chain_id))

    # 统计提取了多少残基
    struct2 = parser.get_structure("chain", out_pdb)
    n_res = sum(1 for _ in struct2.get_residues())
    print(f"  [BioPython] Extracted chain {chain_id} ({n_res} residues) → {out_pdb}")


# ==============================================================================
# Step 3 & 4: PDBFixer 清理 + OpenMM 加氢 + 参数化
# ==============================================================================
def prepare_and_parameterize(pdb_file):
    """
    使用 PDBFixer 清理 PDB，然后用 OpenMM + AMBER ff14SB 参数化。
    
    PDBFixer 处理:
    - 替换非标准残基 (如 MSE → MET)
    - 移除 HETATM (配体、水)
    - 补全缺失的重原子
    
    OpenMM 处理:
    - 按 AMBER ff14SB 力场规范添加氢原子 (pH=7.0)
    - 为每个原子分配部分电荷 (q)、LJ σ、LJ ε
    
    Returns:
        coords:     [N, 3]  原子坐标 (Å)
        atom_types: [N]     元素符号
        atom_names: [N]     原子名 (CA, CB, ...)
        radii:      [N]     vdW 半径 = σ/2 (Å)
        epsilon:    [N]     LJ ε (kJ/mol)
        sigma:      [N]     LJ σ (Å)
        charges:    [N]     部分电荷 (e)
        res_names:  [N]     残基名
        res_ids:    [N]     残基编号
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile, ForceField, Modeller
    from openmm import NonbondedForce
    from openmm import unit

    # --- PDBFixer: 修补 PDB ---
    fixer = PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    # 移除末端缺失残基（通常不可靠，只保留内部的）
    chains = list(fixer.topology.chains())
    keys_to_remove = []
    for key in fixer.missingResidues:
        chain_idx, res_idx = key
        chain = chains[chain_idx]
        chain_residues = list(chain.residues())
        if res_idx == 0 or res_idx == len(chain_residues):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del fixer.missingResidues[key]

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    print("  [PDBFixer] Cleaned PDB (fixed non-standard residues, added missing atoms)")

    # --- OpenMM: 加氢 + 参数化 ---
    forcefield = ForceField("amber14-all.xml")

    # Modeller 加氢（按 AMBER 力场命名规范）
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield, pH=7.0)

    topology = modeller.topology
    positions = modeller.positions

    # 创建力场 System
    system = forcefield.createSystem(topology)

    # 找到 NonbondedForce（存储 LJ 参数和电荷的 Force 对象）
    nonbonded = None
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        raise RuntimeError("No NonbondedForce found in the OpenMM System")

    n_atoms = system.getNumParticles()
    print(f"  [OpenMM] Parameterized with AMBER ff14SB: {n_atoms} atoms")

    # --- 提取所有原子级数据 ---

    # 坐标 (OpenMM 内部用 nm，转为 Å)
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for i, pos in enumerate(positions):
        coords[i, 0] = pos[0].value_in_unit(unit.angstrom)
        coords[i, 1] = pos[1].value_in_unit(unit.angstrom)
        coords[i, 2] = pos[2].value_in_unit(unit.angstrom)

    # 原子信息（从 topology 获取）
    atom_types_list = []
    atom_names_list = []
    res_names_list = []
    res_ids_list = []
    for atom in topology.atoms():
        atom_types_list.append(atom.element.symbol if atom.element else "X")
        atom_names_list.append(atom.name)
        res_names_list.append(atom.residue.name)
        res_ids_list.append(int(atom.residue.id))

    # LJ 参数和电荷（从 NonbondedForce 获取）
    charges = np.zeros(n_atoms, dtype=np.float64)
    sigma = np.zeros(n_atoms, dtype=np.float64)
    epsilon = np.zeros(n_atoms, dtype=np.float64)

    for i in range(n_atoms):
        q, s, e = nonbonded.getParticleParameters(i)
        charges[i] = q.value_in_unit(unit.elementary_charge)
        sigma[i] = s.value_in_unit(unit.nanometer) * 10.0  # nm → Å
        epsilon[i] = e.value_in_unit(unit.kilojoule_per_mole)

    # vdW 半径 = 2.0 ** (1.0 / 6.0)σ / 2
    # (两原子 vdW 接触距离 = σ_i/2 + σ_j/2 = σ_ij)
    radii = (2.0 ** (1.0 / 6.0)) * sigma / 2.0

    return (
        coords,
        np.array(atom_types_list),
        np.array(atom_names_list),
        radii,
        epsilon,
        sigma,
        charges,
        np.array(res_names_list),
        np.array(res_ids_list, dtype=np.int32),
    )


# ==============================================================================
# Main
# ==============================================================================

# ==============================================================================
# 处理单个 PDB 文件
# ==============================================================================
def process_one_pdb(pdb_file, chain_id, output_dir):
    """
    处理单个 PDB 文件的完整流程。

    Args:
        pdb_file: PDB 文件路径
        chain_id: 要提取的链 ID
        output_dir: 输出目录

    Returns:
        (prefix, n_atoms) on success, raises Exception on failure
    """
    pdb_id = os.path.splitext(os.path.basename(pdb_file))[0]
    prefix = f"{pdb_id}_{chain_id}"

    # 每个 PDB 输出到以 pdb_id 命名的子文件夹
    pdb_output_dir = os.path.join(output_dir, pdb_id)
    os.makedirs(pdb_output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: reduce 去氢
        trimmed_pdb = os.path.join(tmp_dir, "trimmed.pdb")
        remove_hydrogens_with_reduce(pdb_file, trimmed_pdb)

        # Step 2: 提取单链
        chain_pdb = os.path.join(tmp_dir, "chain.pdb")
        extract_chain(trimmed_pdb, chain_pdb, chain_id)

        # Step 3: PDBFixer 清理 + OpenMM 加氢 + AMBER 参数化
        (
            coords,
            atom_types,
            atom_names,
            radii,
            epsilon,
            sigma,
            charges,
            res_names,
            res_ids,
        ) = prepare_and_parameterize(chain_pdb)

    # 保存结果
    outputs = {
        "coords": coords,
        "atom_types": atom_types,
        "atom_names": atom_names,
        "radii": radii,
        "epsilon": epsilon,
        "sigma": sigma,
        "charges": charges,
        "res_names": res_names,
        "res_ids": res_ids,
    }
    for key, arr in outputs.items():
        path = os.path.join(pdb_output_dir, f"{prefix}_{key}.npy")
        np.save(path, arr)

    # 打印摘要
    print(f"  {'─'*50}")
    print(f"  {prefix}: {len(coords)} atoms")
    print(f"  Charge:  [{charges.min():+.4f}, {charges.max():+.4f}] e, "
          f"net={charges.sum():+.2f} e")
    print(f"  Sigma:   [{sigma.min():.3f}, {sigma.max():.3f}] Å")
    print(f"  Epsilon: [{epsilon.min():.4f}, {epsilon.max():.4f}] kJ/mol")
    unique, counts = np.unique(atom_types, return_counts=True)
    elem_str = ", ".join(f"{e}={c}" for e, c in zip(unique, counts))
    print(f"  Elements: {elem_str}")

    return prefix, len(coords)


# ==============================================================================
# Main
# ==============================================================================
def main():
    # ---- 解析参数 ----
    if len(sys.argv) < 3:
        print(f"用法:")
        print(f"  单文件:  python {sys.argv[0]} <pdb_file> <chain_id> [output_dir]")
        print(f"  批量:    python {sys.argv[0]} --dir <pdb_dir> <chain_id> [output_dir]")
        print(f"示例:")
        print(f"  python {sys.argv[0]} 1MBN.pdb A ./output/")
        print(f"  python {sys.argv[0]} --dir ./raw_pdbs/ A ./output/")
        sys.exit(1)

    # ---- 判断模式 ----
    if sys.argv[1] == "--dir":
        # 批量模式: --dir <pdb_dir> <chain_id> [output_dir]
        if len(sys.argv) < 4:
            print("Error: --dir 模式需要至少 3 个参数: --dir <pdb_dir> <chain_id>")
            sys.exit(1)
        pdb_dir = sys.argv[2]
        chain_id = sys.argv[3]
        output_dir = sys.argv[4] if len(sys.argv) > 4 else "."

        if not os.path.isdir(pdb_dir):
            print(f"Error: 目录不存在: {pdb_dir}")
            sys.exit(1)

        # 收集所有 .pdb 文件
        pdb_files = sorted([
            os.path.join(pdb_dir, f)
            for f in os.listdir(pdb_dir)
            if f.lower().endswith(".pdb")
        ])

        if not pdb_files:
            print(f"Error: 目录中没有找到 .pdb 文件: {pdb_dir}")
            sys.exit(1)

        print(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
        print(f"Chain: {chain_id}, Output: {output_dir}")
        print(f"{'='*60}\n")

        succeeded = []
        failed = []

        for i, pdb_file in enumerate(pdb_files, 1):
            pdb_name = os.path.basename(pdb_file)
            print(f"[{i}/{len(pdb_files)}] Processing {pdb_name} chain {chain_id} ...")
            try:
                prefix, n_atoms = process_one_pdb(pdb_file, chain_id, output_dir)
                succeeded.append((pdb_name, prefix, n_atoms))
                print(f"  ✓ Done\n")
            except Exception as e:
                failed.append((pdb_name, str(e)))
                print(f"  ✗ FAILED: {e}\n")

        # ---- 批量摘要 ----
        print(f"\n{'='*60}")
        print(f"  BATCH SUMMARY")
        print(f"{'='*60}")
        print(f"  Total:     {len(pdb_files)}")
        print(f"  Succeeded: {len(succeeded)}")
        print(f"  Failed:    {len(failed)}")

        if succeeded:
            print(f"\n  Succeeded:")
            for pdb_name, prefix, n_atoms in succeeded:
                print(f"    {pdb_name:30s} → {prefix} ({n_atoms} atoms)")

        if failed:
            print(f"\n  Failed:")
            for pdb_name, err in failed:
                print(f"    {pdb_name:30s} → {err}")

        print(f"\n  Output directory: {os.path.abspath(output_dir)}")
        print(f"{'='*60}")

    else:
        # 单文件模式: <pdb_file> <chain_id> [output_dir]
        pdb_file = sys.argv[1]
        chain_id = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "."

        if not os.path.exists(pdb_file):
            print(f"Error: PDB file not found: {pdb_file}")
            sys.exit(1)

        print(f"Processing {pdb_file}, chain {chain_id}")
        print(f"{'='*60}")
        process_one_pdb(pdb_file, chain_id, output_dir)
        print(f"\n  Output directory: {os.path.abspath(output_dir)}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
