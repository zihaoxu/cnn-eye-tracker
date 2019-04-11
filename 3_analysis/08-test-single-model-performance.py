# Load in all dependencies and helper functions from the main folder
import sys
sys.path.append('..//')
from utils import *

# Set seed
np.random.seed(47)

# Helper func
def mae(array):
	return str(round(np.mean(abs(array)),8))

# Load in validation set and model
suffix = '_mac'
BATCH_SIZE = 32
INIT_LR = 5e-6
DROPOUT = 30
X_val = pickle_loader(['X_val'+suffix])
df = pd.read_csv(MST_PATH + 'df_val'+suffix+'.csv')

# Load the best model and predict
suffix += "_clip"
model_name = "mobile(mae)_adam_"+str(BATCH_SIZE)+"_"+str(INIT_LR)+suffix
base_filepath = MODEL_SAVE_PATH + model_name + ".hdf5"
print("Loading model:", model_name)
model = load_model(base_filepath)
val_pred_xy = model.predict(X_val, batch_size=BATCH_SIZE, verbose=1)

# Setup folder to save output
SAVE_PATH = MODEL_PERFORM_PATH + model_name + "//"
if not os.path.isdir(SAVE_PATH):
	os.mkdir(MODEL_PERFORM_PATH + model_name)

# Append prediction to df_val
df['coord_x'] = np.clip(df['coord_x'], 0, 1)
df['coord_y'] = np.clip(df['coord_y'], 0, 1)
df['val_pred_x'] = val_pred_xy[0]
df['val_pred_y'] = val_pred_xy[1]
df['val_pred_x'] = np.clip(df['val_pred_x'], 0, 1)
df['val_pred_y'] = np.clip(df['val_pred_y'], 0, 1)
df['residual_x'] = df['coord_x'] - df['val_pred_x']
df['residual_y'] = df['coord_y'] - df['val_pred_y']
df['residual_x'] = df['coord_x'] - df['val_pred_x']
df['residual_y'] = df['coord_y'] - df['val_pred_y']

# Print MAE
print("MAE in x direction:", mae(df['residual_x']), \
      "MAE in y direction:",  mae(df['residual_y']))

# Predicted versus actual scatter
plt.scatter(df['coord_x'], df['coord_y'], s = 1, alpha = 0.2, label = 'true')
plt.scatter(df['val_pred_x'], df['val_pred_y'], s = 1, c = 'r', alpha = 0.2, label = 'pred')
plt.legend()
plt.xlabel("x_coordinate")
plt.ylabel("y_coordinate")
plt.savefig(SAVE_PATH + "true_pred_scatter", dpi = 300)
plt.close()

# Predicted versus actual 45 degree
plt.scatter(df['val_pred_x'], df['coord_x'], s = 1, alpha = 0.1)
plt.xlabel("predicted")
plt.ylabel("actual")
plt.savefig(SAVE_PATH + "true_pred_x", dpi = 300)
plt.close()
plt.scatter(df['val_pred_y'], df['coord_y'], s = 1, alpha = 0.1)
plt.xlabel("predicted")
plt.ylabel("actual")
plt.savefig(SAVE_PATH + "true_pred_y", dpi = 300)
plt.close()

# Preparing webgazer predictions
df['webGazerX'] = np.clip(df['webGazerX'], 0, 1)
df['webGazerY'] = np.clip(df['webGazerY'], 0, 1)
df['webgazer_residual_x'] = df['coord_x'] - df['webGazerX']
df['webgazer_residual_y'] = df['coord_y'] - df['webGazerY']

# Predicted versus actual scatter WebGazer
plt.scatter(df['coord_x'], df['coord_y'], s = 1, alpha = 0.2, label = 'true')
plt.scatter(df['webGazerX'], df['webGazerY'], s = 1, c = 'r', alpha = 0.2, label = 'pred')
plt.legend()
plt.xlabel("x_coordinate")
plt.ylabel("y_coordinate")
plt.savefig(SAVE_PATH + "true_pred_scatter_webgazer", dpi = 300)
plt.close()

# Comparison with web gazer
sns.distplot(df['webgazer_residual_x'], bins = 50, label = 'WebGazer')
sns.distplot(df['residual_x'],bins = 50, label = 'CNN')
plt.legend()
plt.savefig(SAVE_PATH + "webgazer_cnn_x", dpi = 300)
plt.close()
sns.distplot(df['webgazer_residual_y'], bins = 50, label = 'WebGazer')
sns.distplot(df['residual_y'],bins = 50, label = 'CNN')
plt.legend()
plt.savefig(SAVE_PATH + "webgazer_cnn_y", dpi = 300)
plt.close()

# Define a single metric for both models:
df['webGazer_score'] = (df['webgazer_residual_x']**2 + df['webgazer_residual_y']**2)**0.5
df['CNN_score'] = (df['residual_x']**2 + df['residual_y']**2)**0.5
sns.distplot(df['webGazer_score'], bins = 50, label = 'WebGazer')
sns.distplot(df['CNN_score'],bins = 50, label = 'CNN')
plt.legend()
plt.savefig(SAVE_PATH + "webgazer_cnn_all", dpi = 300)
plt.close()

# Save model performance
print("Save model performance...")
write_model_performance(model_name, 
	["MAE_x:" + mae(df['residual_x']),
	"MAE_y:" + mae(df['residual_y']),
	"MAE_avg:" + mae(np.append(df['residual_x'], df['residual_y'])),
	"MAE_euclidean:" + str(np.mean(df['CNN_score'])),
	"MAE_x_wg:" + mae(df['webgazer_residual_x']),
	"MAE_y_wg:" + mae(df['webgazer_residual_y']),
	"MAE_avg_wg:" + mae(np.append(df['webgazer_residual_x'], df['webgazer_residual_y'])),
	"MAE_euclidean_wg:" + str(np.mean(df['webGazer_score'])),
	"CNN_better_all:" + str(np.mean(df['CNN_score'] < df['webGazer_score'])),
	"CNN_better_x:" + str(np.mean(df['residual_x'] < df['webgazer_residual_x'])),
	"CNN_better_y:" + str(np.mean(df['residual_y'] < df['webgazer_residual_y']))
	])