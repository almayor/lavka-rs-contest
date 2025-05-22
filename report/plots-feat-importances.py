import matplotlib.pyplot as plt
import pandas as pd

feature_importances = {
    "session_duration_seconds": 34.699214,
    "source_type": 28.756725,
    "session_unique_products": 11.094275,
    "session_length": 1.968416,
    "session_unique_stores": 1.026158,
    "bpr_popular_score": 0.773342,
    "purchases_view_ratio_u_p_1y": 0.733973,
    "product_total_purchases": 0.582718,
    "product_embed_1": 0.564106,
    "purchases_view_ratio_u_s_1y": 0.546431,
    "hour_of_day_sin": 0.460576,
    "cat_embed_18": 0.453753,
    "product_embed_12": 0.431691,
    "puresvd_cf_score": 0.419249,
    "product_embed_5": 0.395379,
    "purchases_view_ratio_u_c_source_1y": 0.394632,
    "hour_of_day_cos": 0.377454,
    "purchases_u_p_6mo": 0.375394,
    "views_u_p_source_1y": 0.370813,
    "purchases_view_ratio_u_c_1y": 0.366052,
    "cat_std_purchase_hour": 0.356035,
    "cat_embed_2": 0.355639,
    "cat_embed_1": 0.355326,
    "cf_score": 0.351490,
    # ... (rest of the features from the prompt) ...
    "purchases_view_ratio_u_p_source_3mo": 0.000000
}

# Sort features by importance
sorted_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)

# Select top N features
top_n = 20
top_features = dict(sorted_features[:top_n])

# Create bar plot
plt.figure(figsize=(10, 8))
plt.barh(list(top_features.keys()), list(top_features.values()))
plt.xlabel("Feature Importance")
plt.title(f"Top {top_n} Feature Importances")
plt.gca().invert_yaxis() # Display most important at the top
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.show()
