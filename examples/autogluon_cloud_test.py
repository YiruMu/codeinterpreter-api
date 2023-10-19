from autogluon.cloud import TabularCloudPredictor
import pandas as pd

'''
TabularCloudPredictor.generate_trust_relationship_and_iam_policy_file(
                account_id="644385875248",  # The AWS account ID you plan to use for CloudPredictor.
                cloud_output_bucket="codeinterpreter"  # S3 bucket name where intermediate artifacts will be uploaded and trained models should be saved. You need to create this bucket beforehand.
            )
'''

train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
test_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
predictor_init_args = {"label": "class"}  # init args you would pass to AG TabularPredictor
predictor_fit_args = {"train_data": train_data, "time_limit": 120}  # fit args you would pass to AG TabularPredictor
cloud_predictor = TabularCloudPredictor(cloud_output_path='s3://codeinterpreter/autogluon-cloud-poc/')
cloud_predictor.fit(predictor_init_args=predictor_init_args, predictor_fit_args=predictor_fit_args)
cloud_predictor.deploy()
result = cloud_predictor.predict_real_time(test_data)
cloud_predictor.cleanup_deployment()
# Batch inference
#result = cloud_predictor.predict(test_data)
print(result)



