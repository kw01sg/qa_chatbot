# Question Answering Chatbot

Questioning answering chatbot using a knowledge base created from PDF files of financial reports in the `./data` directory. 

Three types of models are available:

* ExtractiveTextTableQAModel
* ExtractiveTextQAModel
* GenerativeTextQAModel

Details on their implementation can be found in [flask_app\model\declarations.py](flask_app\model\declarations.py).


## Create Environment using Conda

```bash
$ # create conda environment
$ conda env create --name <conda_env> python=3.8
$ conda activate <conda_env>
$ pip install -r requirements.txt
$ pip install farm-haystack[faiss,pdf]
```

## Serve Flask App

### Using Flask Built-in Development Server

```bash
$ export FLASK_APP=flask_app
$ flask run --host=0.0.0.0 --port=5000
```

## Docker

### Build and Run Flask App using Docker

* Current image is only CPU compatible
* Update dockerfile, replacing `<app_name>`
* Add `instance/config` file
* Build and Run Image

```bash
$ docker build -t <app_name> -f DockerFile .
$ docker run -d --rm -p 5000:5000 <app_name>

$ # tear down container
$ docker stop <app_name>
```
