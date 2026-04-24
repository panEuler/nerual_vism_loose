def get_logger():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("biomol_surface_unsup")