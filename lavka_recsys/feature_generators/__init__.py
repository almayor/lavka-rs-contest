from .collaborative_filtering import register_cf_features
from .text_processor import register_text_embedding_features
from .common import register_common_features

def register_all_features():
    """
    Register all feature generators.
    """
    register_cf_features()
    register_text_embedding_features()
    register_common_features()