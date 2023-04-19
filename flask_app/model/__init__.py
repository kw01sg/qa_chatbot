import logging
import time

from flask_app.model.declarations import (ExtractiveTextQAModel,
                                          ExtractiveTextTableQAModel,
                                          GenerativeTextQAModel)

model_store = {}


def init_model_store(app):
    logging.info('Loading model store...')
    start_time = time.time()

    model_store['extractive_text'] = ExtractiveTextQAModel()
    model_store['extractive_text'].init_model(app.config)

    # model_store['extractive_text_table'] = ExtractiveTextTableQAModel()
    # model_store['extractive_text_table'].init_model(app.config)

    # model_store['generative_text'] = GenerativeTextQAModel()
    # model_store['generative_text'].init_model(app.config)

    end_time = time.time()

    logging.info('Total time taken to load model store: %.3fs.' %
                 (end_time - start_time))
