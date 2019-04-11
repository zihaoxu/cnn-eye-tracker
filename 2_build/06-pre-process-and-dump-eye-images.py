# Load in all dependencies and helper functions from the main folder
import sys
sys.path.append('..//')
from utils import *

# Set seed
np.random.seed(47)

# Load in csvs and merge
df = pd.read_csv(MST_PATH + 'mst_eye_label.csv')
p_id = pd.read_csv(MST_PATH + 'all_img_facial_lankmarks.csv')
char = pd.read_csv(MST_PATH + 'participant_characteristics.csv')[['setting', 'participant_id']]
df = pd.merge(df, p_id[['participant', 'id', 'task', 'webGazerX', 'webGazerY']], on = 'id', how = 'left')

# Map setting from char to df
char.columns = ['participant' if x == 'participant_id' else x for x in char.columns]
char = char[char['participant'].isin(df['participant'].unique())]
df = pd.merge(df, char, on = 'participant', how = 'left')

# Create the participant id groups
pc_lst = ['P_10', 'P_16', 'P_25']
mac_lst = ['P_39', 'P_52', 'P_64']
pc_rest_id = [i for i in df[df['setting']=='PC']['participant'].unique() if i not in pc_lst]
mac_rest_id = [i for i in df[df['setting']!='PC']['participant'].unique() if i not in mac_lst]
print("pc_lst:", pc_lst)
print("mac_lst:", mac_lst)
print("pc_rest_id:", pc_rest_id)
print("mac_rest_id:", mac_rest_id)
# pc_lst: ['P_10', 'P_16', 'P_25']
# mac_lst: ['P_39', 'P_52', 'P_64']
# pc_rest_id: ['P_40', 'P_06', 'P_41', 'P_42', 'P_45', 'P_46', 'P_27', 'P_55', 'P_56', 'P_31', 'P_59', 'P_08', 'P_12', 'P_13', 'P_17', 'P_18', 'P_19', 'P_35', 'P_20', 'P_23']
# mac_rest_id: ['P_01', 'P_47', 'P_50', 'P_51', 'P_07', 'P_02', 'P_53', 'P_54', 'P_57', 'P_24', 'P_29', 'P_58', 'P_60', 'P_63', 'P_14', 'P_15', 'P_33', 'P_37', 'P_38']

# Create and shuffle train and val sets
df_train_pc = df[df['participant'].isin(pc_rest_id)].sample(frac=1).reset_index(drop=True)
df_train_mac = df[df['participant'].isin(mac_rest_id)].sample(frac=1).reset_index(drop=True)
df_val_pc = df[df['participant'].isin(pc_lst)].sample(frac=1).reset_index(drop=True)
df_val_mac = df[df['participant'].isin(mac_lst)].sample(frac=1).reset_index(drop=True)

# Save train and test to MST
df_train_pc.to_csv(MST_PATH + 'df_train_pc.csv', index = None)
df_train_mac.to_csv(MST_PATH + 'df_train_mac.csv', index = None)
df_val_pc.to_csv(MST_PATH + 'df_val_pc.csv', index = None)
df_val_mac.to_csv(MST_PATH + 'df_val_mac.csv', index = None)

# Drop irrelevant columns
df_train_pc = df_train_pc.drop(['id', 'participant', 'task'], 1)
df_train_mac = df_train_mac.drop(['id', 'participant', 'task'], 1)
df_val_pc = df_val_pc.drop(['id', 'participant', 'task'], 1)
df_val_mac = df_val_mac.drop(['id', 'participant', 'task'], 1)
print(df_train_pc.shape)
print(df_train_mac.shape)
print(df_val_pc.shape)
print(df_val_mac.shape)

del df, p_id
gc.collect()

# Load in data physically
def df_train_pickle_saver(df_train, suffix):
	slices = 	[(0, len(df_train)//3),
				(len(df_train)//3+1, 2*len(df_train)//3),
				(2*len(df_train)//3+1, len(df_train))]
	for batch_n in range(3):
		print("image_loader train "+ str(batch_n) + '_' + suffix +" begins...")
		start_ind, end_ind = slices[batch_n]
		X_train, y_train_coordx, y_train_coordy = image_loader(df_train.loc[start_ind:end_ind,])
		# Dump pickle files
		pickle_name = '_' + str(batch_n) + '_' + suffix + '.pickle'
		print("Dumping train "+ pickle_name + " begins...")
		with open(PICKLE_PATH+'X_train'+pickle_name,'wb') as f:
		    pickle.dump(X_train, f, protocol=pickle.HIGHEST_PROTOCOL)
		with open(PICKLE_PATH+'y_train_coordx'+pickle_name,'wb') as f:
		    pickle.dump(y_train_coordx, f, protocol=pickle.HIGHEST_PROTOCOL)
		with open(PICKLE_PATH+'y_train_coordy'+pickle_name,'wb') as f:
		    pickle.dump(y_train_coordy, f, protocol=pickle.HIGHEST_PROTOCOL)

		del X_train, y_train_coordx, y_train_coordy
		gc.collect()

def df_val_pickle_saver(df_val, suffix):
	print("image_loader val_" + suffix + " begins...")
	X_val, y_val_coordx, y_val_coordy = image_loader(df_val)
	del df_val
	gc.collect()

	with open(PICKLE_PATH+'X_val_'+suffix+'.pickle','wb') as f:
	    pickle.dump(X_val, f, protocol=pickle.HIGHEST_PROTOCOL)
	with open(PICKLE_PATH+'y_val_coordx_'+suffix+'.pickle','wb') as f:
	    pickle.dump(y_val_coordx, f, protocol=pickle.HIGHEST_PROTOCOL)
	with open(PICKLE_PATH+'y_val_coordy_'+suffix+'.pickle','wb') as f:
	    pickle.dump(y_val_coordy, f, protocol=pickle.HIGHEST_PROTOCOL)

df_train_pickle_saver(df_train_pc, 'pc')
df_train_pickle_saver(df_train_mac, 'mac')
df_val_pickle_saver(df_val_pc, 'pc')
df_val_pickle_saver(df_val_mac, 'mac')

# Small test
# with open(PICKLE_PATH+'X_train.pickle', 'rb') as f:
#     X_train = pickle.load(f)
# print(X_train)
