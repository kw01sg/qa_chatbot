import logging
from abc import ABC, abstractmethod
from pathlib import Path

from haystack import Answer, Pipeline
from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.nodes import (BM25Retriever, DensePassageRetriever, FARMReader,
                            JoinAnswers, RAGenerator, RouteDocuments,
                            TableReader)
from haystack.pipelines import ExtractiveQAPipeline, GenerativeQAPipeline

from utils.data import pdf_to_text_and_tables, preprocess_text_documents


class BaseModel(ABC):
    def __init__(self):
        self.predictor = None

    @abstractmethod
    def init_model(self, config):
        pass

    @abstractmethod
    def predict(self, input):
        pass

    @abstractmethod
    def format_prediction(self, prediction):
        pass


class ExtractiveTextTableQAModel(BaseModel):
    def init_model(self, config):
        if 'DATA_DIR' not in config:
            # TODO: raise a custom InitModel exception
            logging.error("data_dir not in config, unable to init model")
            return

        data_dir = Path(config.get('DATA_DIR'))
        text_list, table_list = self.prepare_data(data_dir)
        self.document_store = self.init_document_store(text_list, table_list)

        # init retriever
        retriever = BM25Retriever(document_store=self.document_store)

        # init readers
        text_reader = FARMReader("deepset/roberta-base-squad2", use_gpu=False)
        table_reader = TableReader("deepset/tapas-large-nq-hn-reader",
                                   use_gpu=False)

        # other nodes
        route_documents = RouteDocuments()
        join_answers = JoinAnswers()

        # pipeline
        self.text_table_qa_pipeline = Pipeline()
        self.text_table_qa_pipeline.add_node(component=retriever,
                                             name="BM25Retriever",
                                             inputs=["Query"])
        self.text_table_qa_pipeline.add_node(component=route_documents,
                                             name="RouteDocuments",
                                             inputs=["BM25Retriever"])
        self.text_table_qa_pipeline.add_node(component=text_reader,
                                             name="TextReader",
                                             inputs=["RouteDocuments.output_1"])
        self.text_table_qa_pipeline.add_node(component=table_reader,
                                             name="TableReader",
                                             inputs=["RouteDocuments.output_2"])
        self.text_table_qa_pipeline.add_node(component=join_answers,
                                             name="JoinAnswers",
                                             inputs=["TextReader", "TableReader"])

    def prepare_data(self, data_dir: Path):
        text_list, table_list = pdf_to_text_and_tables(data_dir,
                                                       read_tables=True)
        processed_text_list = preprocess_text_documents(text_list)

        return processed_text_list, table_list

    def init_document_store(self, text_list, table_list):
        # DocumentStore stores the Documents that the question answering
        # system uses to find answers to your questions
        document_store = InMemoryDocumentStore(use_bm25=True)
        document_store.write_documents(text_list,
                                       duplicate_documents='skip')
        document_store.write_documents(table_list)

        return document_store

    def predict(self, input):
        return self.text_table_qa_pipeline.run(query=input)

    def format_prediction(self, prediction):
        # get the best answer as well as its score
        answer: Answer = prediction['answers'][0]
        return answer.answer, answer.score


class ExtractiveTextQAModel(BaseModel):
    def init_model(self, config):
        if 'DATA_DIR' not in config:
            # TODO: raise a custom InitModel exception
            logging.error("data_dir not in config, unable to init model")
            return

        data_dir = Path(config.get('DATA_DIR'))
        text_list, _ = self.prepare_data(data_dir)
        self.document_store = self.init_document_store(text_list)

        # init retriever
        retriever = BM25Retriever(document_store=self.document_store)

        # init readers
        text_reader = FARMReader("deepset/roberta-base-squad2", use_gpu=False)

        # pipeline
        self.text_qa_pipeline = ExtractiveQAPipeline(text_reader, retriever)

    def prepare_data(self, data_dir: Path):
        text_list, _ = pdf_to_text_and_tables(data_dir)
        processed_text_list = preprocess_text_documents(text_list)

        return processed_text_list, _

    def init_document_store(self, text_list):
        # DocumentStore stores the Documents that the question answering
        # system uses to find answers to your questions
        document_store = InMemoryDocumentStore(use_bm25=True)
        document_store.write_documents(text_list,
                                       duplicate_documents='skip')

        return document_store

    def predict(self, input):
        return self.text_qa_pipeline.run(
            query=input,
            params={"Retriever": {"top_k": 10},
                    "Reader": {"top_k": 5}}
        )

    def format_prediction(self, prediction):
        # get the best answer as well as its score
        answer: Answer = prediction['answers'][0]
        return answer.answer, answer.score


class GenerativeTextQAModel(BaseModel):
    def init_model(self, config):
        if 'DATA_DIR' not in config:
            # TODO: raise a custom InitModel exception
            logging.error("data_dir not in config, unable to init model")
            return

        data_dir = Path(config.get('DATA_DIR'))
        text_list, _ = self.prepare_data(data_dir)
        self.document_store = self.init_document_store(text_list)

        # init retriever
        retriever = DensePassageRetriever(
            document_store=self.document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            use_gpu=False,
            embed_title=True,
        )

        # Add documents embeddings to index
        self.document_store.update_embeddings(retriever=retriever)

        # init generator
        generator = RAGenerator(
            model_name_or_path="facebook/rag-token-nq",
            use_gpu=False,
            top_k=1,
            max_length=200,
            min_length=2,
            embed_title=True,
            num_beams=2,
        )

        # pipeline
        self.generative_pipeline = GenerativeQAPipeline(generator=generator,
                                                        retriever=retriever)

    def prepare_data(self, data_dir: Path):
        text_list, _ = pdf_to_text_and_tables(data_dir)
        processed_text_list = preprocess_text_documents(text_list)

        return processed_text_list, _

    def init_document_store(self, text_list):
        # DocumentStore stores the Documents that the question answering
        # system uses to find answers to your questions
        document_store = FAISSDocumentStore(faiss_index_factory_str="Flat",
                                            return_embedding=True)
        document_store.write_documents(text_list,
                                       duplicate_documents='skip')

        return document_store

    def predict(self, input):
        return self.generative_pipeline.run(
            query=input,
            params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}}
        )

    def format_prediction(self, prediction):
        # get the best answer as well as its score
        answer: Answer = prediction['answers'][0]
        return answer.answer, answer.score
