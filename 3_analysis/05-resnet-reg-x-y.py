# Load in all dependencies and helper functions from the main folder
import sys
sys.path.append('..//')
from utils import *

# Load in train and validation sets
suffix = '_mac'
pickle_num = 3
X_train = pickle_loader(['X_train_'+str(i)+suffix for i in range(pickle_num)])
y_train_x = pickle_loader(['y_train_coordx_'+str(i)+suffix for i in range(pickle_num)])
y_train_y = pickle_loader(['y_train_coordy_'+str(i)+suffix for i in range(pickle_num)])
X_val = pickle_loader(['X_val'+suffix])
y_val_x = pickle_loader(['y_val_coordx'+suffix])
y_val_y = pickle_loader(['y_val_coordy'+suffix])

# Clipping!
suffix += '_clip'
y_train_x = np.clip(y_train_x, 0, 1)
y_train_y = np.clip(y_train_y, 0, 1)
y_val_x = np.clip(y_val_x, 0, 1)
y_val_y = np.clip(y_val_y, 0, 1)

# create the base pre-trained model
DROPOUT = 0.50
cnn = ResNet50(weights=None, include_top=False, input_shape = (32, 160, 3)) # Need to resize from 20 x 100 to 32 x 160
out = Flatten()(cnn.output)
out = Dropout(DROPOUT)(out)
pred_x = Dense(1, name='pred_x')(out)
pred_y = Dense(1, name='pred_y')(out)

# this is the model we will train
model = Model(inputs=cnn.input, outputs=[pred_x, pred_y])

# Define model params
NUM_EPOCHS = 100
BATCH_SIZE = 64
INIT_LR = 1e-5
PATIENCE = 15
model_name = "resnet50_mae_adam_"+str(BATCH_SIZE)+"_"+\
             str(INIT_LR)+"_"+str(int(DROPOUT*100))+suffix
base_filepath = MODEL_SAVE_PATH + model_name + ".hdf5"

# Save model specs
write_model_specs(model_name, 
	["MODEL_NAME:" + model_name,
	"NUM_EPOCHS:" + str(NUM_EPOCHS),
	"BATCH_SIZE:" + str(BATCH_SIZE),
	"INIT_LR:" + str(INIT_LR),
	"PATIENCE:" + str(PATIENCE),
	"OPTIMIZER:" + str("Adam"),
	"DROPOUT: " + str(DROPOUT)])

# Define adam optimizer
adam = Adam(lr=INIT_LR) 

# Set up early_stopping_monitor and learning_rate_scheduler
early_stopping_monitor = EarlyStopping(monitor = 'val_loss', patience = PATIENCE)
checkpoint = ModelCheckpoint(base_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [early_stopping_monitor, checkpoint] 
    
# compile the model
model.compile(loss = "mean_absolute_error",
               optimizer = adam,
               loss_weights=[.5, .5])

np.random.seed(47)
history = model.fit(X_train, [y_train_x, y_train_y],
			          batch_size=BATCH_SIZE,
			          epochs=NUM_EPOCHS,
			          verbose=1,
			          callbacks = callbacks,
			          validation_data=(X_val, [y_val_x, y_val_y]))

# list all data in history
print(history.history.keys())
with open(PICKLE_PATH + model_name + '.pickle','wb') as f:
    pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(MODEL_VIZ_PATH + model_name, dpi = 300)
plt.show()