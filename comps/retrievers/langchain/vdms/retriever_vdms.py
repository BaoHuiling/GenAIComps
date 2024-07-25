# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime
from dateparser.search import search_dates
import os
import time


from langsmith import traceable
from langchain_community.vectorstores import VDMS # type: ignore
from langchain_community.vectorstores.vdms import VDMS_Client # type: ignore

from config import COLLECTION_NAME, VDMS_HOST, VDMS_PORT, MEANCLIP_CFG

from comps import (
    SearchedDocMetadata,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from meanclip import setup_meanclip_model, MeanCLIPEmbeddings

meanclip_embedding_endpoint = os.getenv("MEANCLIP_EMBEDDING_ENDPOINT", "")

def update_db(prompt, constraints):
    base_date = datetime.datetime.today()
    today_date= base_date.date()
    dates_found =search_dates(prompt, settings={'PREFER_DATES_FROM': 'past', 'RELATIVE_BASE': base_date})
    # if no date is detected dates_found should return None
    if dates_found != None:
        # Print the identified dates
        # print("dates_found:",dates_found)
        for date_tuple in dates_found:
            date_string, parsed_date = date_tuple
            #print(f"Found date: {date_string} -> Parsed as: {parsed_date}")
            date_out = str(parsed_date.date())
            time_out = str(parsed_date.time())
            hours, minutes, seconds = map(float, time_out.split(":"))
            year, month, day_out = map(int, date_out.split("-"))

        # print("today's date", base_date)
        rounded_seconds = min(round(parsed_date.second + 0.5),59)
        parsed_date = parsed_date.replace(second=rounded_seconds, microsecond=0)

        # Convert the localized time to ISO format
        iso_date_time = parsed_date.isoformat()
        iso_date_time = str(iso_date_time)

        if date_string == 'today':
            constraints = {"date": [ "==", date_out]}
        elif date_out != str(today_date) and time_out =='00:00:00': ## exact day (example last firday)
            constraints = {"date": [ "==", date_out]}
        elif date_out == str(today_date) and time_out =='00:00:00': ## when search_date interprates words as dates output is todays date + time 00:00:00
            pass
        else: ## Interval  of time:last 48 hours, last 2 days,..
            constraints = {"date_time": [ ">=", {"_date":iso_date_time}]}

    return constraints

@register_microservice(
    name="opea_service@retriever_vdms",
    service_type=ServiceType.RETRIEVER,
    endpoint="/v1/retrieval",
    host="0.0.0.0",
    port=7000,
)
@traceable(run_type="retriever")
@register_statistics(names=["opea_service@retriever_vdms"])
def retrieve(input: TextDoc) -> SearchedDocMetadata:
    start = time.time()
    constraints = None
    # Only similarity search is supported
    constraints = update_db(input.text, constraints)
    search_res = vector_db.similarity_search_with_score(query=input.text, k=3, filter=constraints)
    print("search_res", search_res)
    searched_docs = []
    metadata_list = []
    score_list = []
    for r, score in search_res:
        searched_docs.append(TextDoc(text=r.page_content))
        metadata_list.append(r.metadata)
        score_list.append(score) # TODO: place here for potential usage
    # print("searched_docs", searched_docs)
    # print("metadata_list", metadata_list)
    # print("score_list", score_list)

    result = SearchedDocMetadata(retrieved_docs=searched_docs, metadata=metadata_list, initial_query=input.text, score=score_list)
    statistics_dict["opea_service@retriever_vdms"].append_latency(time.time() - start, None)
    return result


if __name__ == "__main__":

    # create embeddings using local embedding model
    if meanclip_embedding_endpoint != "":
        # TODO: create embeddings microservice
        # embedder = HuggingFaceHubEmbeddings(model=meanclip_embedding_endpoint)
        pass
    else:
        model, _ = setup_meanclip_model(MEANCLIP_CFG, device="cpu")
        embedder = MeanCLIPEmbeddings(model=model)

    # Create vdms client
    client = VDMS_Client(host=VDMS_HOST, port=VDMS_PORT)
    
    # Create vectorstore
    vector_db = VDMS(
                      client=client, 
                      embedding = embedder,
                      collection_name = COLLECTION_NAME,
                      engine = "FaissFlat",
                      distance_strategy="IP"
    )

    opea_microservices["opea_service@retriever_vdms"].start()
