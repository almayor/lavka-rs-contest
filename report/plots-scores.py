import matplotlib.pyplot as plt
import numpy as np

experiments = {
    # "Random": (0.29347, 0.30071),
    # "Popularity": (0.32899, 0.33188),
    "Classifier (baseline)": (0.41192, 0.41099),
    "Ranker (baseline)": (0.41585, 0.41611),
    "Ranker +feat_select (-28% features)": (0.41621, 0.41715),
    "Ranker +feat_select (-47% features)": (0.41550, 0.41572),
    "Ranker –text feats": (0.41389, 0.41360),
    "Ranker –text_feats +feat_select": (0.41536, 0.41477),
    "Ranker –session_feats": (0.41856, 0.41776),
    "Ranker -session_feats +clean_target": (0.41953, 0.41735),
    "Ranker -session_feats +clean_target +hparam_opt": (0.41945, 0.41875),
    "Ranker -session_feats +clean_target +hparam_opt +feat_select": (0.41769, 0.41661),
    "Ranker -session_feats +clean_target +hparam_opt +weigh_posinreq": (0.41589, 0.41945),
}

public_scores  = [score[0] for score in experiments.values()]
private_scores = [score[1] for score in experiments.values()]
experiment_labels = list(experiments.keys())

y = np.arange(len(experiments)) # y-positions for the bars
bar_h = 0.35 # height of the bars

fig, ax = plt.subplots(figsize=(10, 8)) # adjusted figure size for better label visibility

ax.barh(y + bar_h/2, public_scores, height=bar_h, label="Public", color='skyblue')
ax.barh(y - bar_h/2, private_scores, height=bar_h, label="Private", color='salmon')

ax.set_xlabel("NDCG@10")
ax.set_title("Kaggle Leaderboard Scores")
ax.set_yticks(y)
ax.set_yticklabels(experiment_labels, fontsize=10)
ax.set_xlim(left=0.41, right=0.42) # adjust as needed 

# Add vertical gridlines
ax.grid(axis='x', linestyle='--', alpha=0.7)

# invert y-axis to have the first experiment at the top
ax.invert_yaxis()

ax.legend()
fig.tight_layout()
plt.savefig('scores.png')
plt.show()
