from codeinterpreterapi import CodeInterpreterSession, File
from autogluon.cloud import TabularCloudPredictor
from langchain.tools import BaseTool
import pandas as pd
from typing import Optional

class AutoGluonTabular(BaseTool):
    name: str = "autogluon_tabular"
    description: str ="""
    This tool uses Autogluon to solve machine learning problems that use tabular data. 
    It will take in exactly 3 arguments: the label the user wants to predict, train_dataset, and the test_dataset."""

    def _run(self, label: Optional[str] = None, train_dataset:Optional[str]=None, test_dataset:Optional[str]=None):
        raise NotImplementedError()

    async def _arun(self, label: Optional[str] = None, train_dataset:Optional[str]=None, test_dataset:Optional[str]=None):
        
        '''
        try:
            TabularCloudPredictor.generate_trust_relationship_and_iam_policy_file(
                account_id="644385875248",  # The AWS account ID you plan to use for CloudPredictor.
                cloud_output_bucket="codeinterpreter"  # S3 bucket name where intermediate artifacts will be uploaded and trained models should be saved. You need to create this bucket beforehand.
            )
        except:
            print("failed to generate_trust_relationship")
            pass 

        '''
        
        try: 
            train_data = pd.read_csv(train_dataset)
            test_data = pd.read_csv(test_dataset)
            predictor_init_args = {"label": label}  # init args you would pass to AG TabularPredictor
            predictor_fit_args = {"train_data": train_data, "time_limit": 30}  # fit args you would pass to AG TabularPredictor
            cloud_predictor = TabularCloudPredictor(cloud_output_path='s3://codeinterpreter/autogluon-cloud-poc/')
            cloud_predictor.fit(predictor_init_args=predictor_init_args, predictor_fit_args=predictor_fit_args)
            cloud_predictor.deploy()
            result = cloud_predictor.predict_real_time(test_data)
            cloud_predictor.cleanup_deployment()
            # Batch inference
            #result = cloud_predictor.predict(test_data)
            return result 
        except Exception as e:
            return "Something went wrong."


async def main():
    # context manager for start/stop of the session

    # continuous session
    async with CodeInterpreterSession(model="gpt-3.5-turbo", additional_tools=[AutoGluonTabular()]) as session:
        while True:
            response = await session.agenerate_response(input("\nUser: "))
            response.show()

    ''' only allow one turn of conversation 
    async with CodeInterpreterSession(additional_tools=[AutoGluonTabular()]) as session:
        # define the user request
        user_request = "train and evaluate the 'class' label using this training dataset: https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv and this test dataset https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv."
        
        response = await session.agenerate_response(user_request)

        # output the response (text + image)
        response.show()
    '''

if __name__ == "__main__": 
    import asyncio

    # run the async function
    asyncio.run(main())
   
