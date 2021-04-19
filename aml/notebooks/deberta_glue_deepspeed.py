from azureml.core import Workspace, Datastore, Dataset, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.runconfig import MpiConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.exceptions import ComputeTargetException

subscription_id = '42ae47bd-b19b-42c1-b0b9-19fd5be9d51b'
resource_group = 'bert-base'
workspace_name = 'SubstrateIntelligenceNLR-WS2'
gpu_cluster_name = "sriovdedicated1"

# subscription_id = 'ea482afa-3a32-437c-aa10-7de928a9e793'
# resource_group = 'onnx_training'
# workspace_name = 'ort_training_dev'
# gpu_cluster_name = "onnx-training-ib"

ws = Workspace(subscription_id, resource_group, workspace_name)
ws_details = ws.get_details()
print('Name:\t\t{}\nLocation:\t{}'.format(ws_details['name'], ws_details['location']))

try:
    gpu_compute_target = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    if gpu_compute_target.provisioning_state == 'Failed':
        gpu_compute_target.delete()
        gpu_compute_target.wait_for_completion(show_output=True)
        raise ComputeTargetException('failed cluster')
    print('Found existing compute target.')
except ComputeTargetException:
    print('cannot find compute target...')
    gpu_compute_target = None

import pprint as pp

if gpu_compute_target is not None:
    pp.pprint(gpu_compute_target.status.serialize())
else:
    raise ValueError("gpu_compute_target not found")

# connecting data stores
ds = Datastore.register_azure_blob_container(workspace=ws,
                                             datastore_name='t1',
                                             container_name='azureml-blobstore-d6fc2475-ad02-44a7-90ff-88a2a91e66b1',
                                             account_name='substrateintel3704284680',
                                             account_key='aHN5vf4m/Hd4ogssPRArAIgBb88IV6eMBSe1HwWYgth9kBcwvVBxtAcPM8XLOhZSq2CBrrvX5gQUMrySKCxpuA==',
                                             create_if_not_exists=True
                                             )

num_nodes = 4
gpus_per_node = 4

# https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.environment(class)?view=azure-ml-py
pytorch_env = Environment(name='hf_deberta_env0')
pytorch_env.docker.enabled = True
pytorch_env.docker.base_image = "hf/transformer/deberta:v0.0.1"
pytorch_env.python.user_managed_dependencies = True
pytorch_env.docker.base_image_registry.address = 'hanyu.azurecr.io'
pytorch_env.docker.base_image_registry.username = 'hanyu'
pytorch_env.docker.base_image_registry.password = 'mhewjKhbjwYAmjtt1IugCYwm3qSKwhA/'
# https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.environment.pythonsection?view=azure-ml-py
pytorch_env.python.interpreter_path = '/opt/miniconda/bin/python'


def get_input_dataset(datastore, path_on_datastore, dataset_name):
    dataset = Dataset.File.from_files(path=[(datastore, path_on_datastore)])
    return dataset.as_named_input(dataset_name).as_mount()


def get_output_dataset(datastore, path_on_datastore, dataset_name):
    return OutputFileDatasetConfig(destination=(datastore, path_on_datastore), name=dataset_name).as_mount()


experiment_name = 'hanyu-deberta-aml-deepspeed-0'
outputSuffix = "_0"

all_params = [
    "--model_name_or_path", "microsoft/deberta-v2-xxlarge",
    "--task_name", "mnli",
    "--do_train",
    "--do_eval",
    "--max_seq_length", 256,
    "--per_device_train_batch_size", 4,
    "--learning_rate", 3e-6,
    "--num_train_epochs", 3,
    "--output_dir", get_output_dataset(ds, "hanyu/deberta/deepspeed/glue_output_dir" + outputSuffix, "output_dir"),
    "--overwrite_output_dir",
    "--logging_steps", 10,
    "--logging_dir", get_output_dataset(ds, "hanyu/deberta/deepspeed/glue_log_dir" + outputSuffix, "logging_dir"),
    # "--cache_dir", get_output_dataset(ds, "hanyu/deberta/deepspeed/cache_dir" + outputSuffix, "cache_dir"),
    "--deepspeed", "aml/ds_config.json",
]

print("creating ScriptRunConfig...")
src = ScriptRunConfig(
    source_directory='../../',
    script='examples/text-classification/run_glue.py',
    arguments=all_params,
    compute_target=gpu_compute_target,
    distributed_job_config=MpiConfiguration(process_count_per_node=gpus_per_node, node_count=num_nodes),
    environment=pytorch_env,
)

print("submitting experiment...")
experiment = Experiment(ws, experiment_name)
run = experiment.submit(src)

print(f"\n{run.get_portal_url()}")
