from .collaborative_filtering import register_cf_fgens
from .common import register_common_fgens
from .text_processor import register_text_embedding_fgens
from .bpr import register_bpr_fgens


def register_all_fgens():
    """
    Register all feature generators.
    """
    register_cf_fgens()
    register_text_embedding_fgens()
    register_common_fgens()
    register_bpr_fgens()
