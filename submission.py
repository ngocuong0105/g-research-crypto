# Example submission. Note the make_env() method can be used once per kerner run. 
# You need to restart to use it again (competition rules). 
#%%
import gresearch_crypto
env = gresearch_crypto.make_env()
iter_test = env.iter_test()
#%%
i = 0 
for (test_df, sample_prediction_df) in iter_test:
    print(i)
    sample_prediction_df['Target'] = 0
    env.predict(sample_prediction_df)
    i+=1

