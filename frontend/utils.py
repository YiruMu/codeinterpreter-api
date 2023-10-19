import os
import shutil
import tempfile
from typing import Optional

import streamlit as st

from codeinterpreterapi import CodeInterpreterSession

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




def create_temp_folder() -> str:
    """
    Creates a temp folder
    """
    temp_folder = tempfile.mkdtemp()
    return temp_folder


async def get_images(prompt: str, files: Optional[list] = None):
    if files is None:
        files = []
    with st.chat_message("user"):  # type: ignore
        st.write(prompt)
    with st.spinner():
        async with CodeInterpreterSession(model="gpt-3.5-turbo") as session:
            response = await session.agenerate_response(prompt, files=files)

            with st.chat_message("assistant"):  # type: ignore
                st.write(response.content)

                # Showing Results
                for _file in response.files:
                    st.image(_file.get_image(), caption=prompt, use_column_width=True)

                # Allowing the download of the results
                if len(response.files) == 1:
                    st.download_button(
                        "Download Results",
                        response.files[0].content,
                        file_name=response.files[0].name,
                    )
                else:
                    target_path = tempfile.mkdtemp()
                    for _file in response.files:
                        _file.save(os.path.join(target_path, _file.name))

                    zip_path = os.path.join(os.path.dirname(target_path), "archive")
                    shutil.make_archive(zip_path, "zip", target_path)

                    with open(zip_path + ".zip", "rb") as f:
                        st.download_button(
                            "Download Results",
                            f,
                            file_name="archive.zip",
                        )