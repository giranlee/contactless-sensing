# Knowledge Structure and Temporal Trends of Contactless Wireless Sensing Research for Health Monitoring in Older Adults: Systematic Search of Abstracts

Giran Lee, Hyowon Kim

**Background:** The growing population of older adults has increased the demand for technologies that enable continuous health monitoring. Contactless wireless sensing can monitor vital signs, detect falls, and assess daily activities using wireless signals without requiring skin-contact sensors or cameras. However, few studies have examined research trends across multiple wireless sensing technologies in the context of health monitoring for older adults.

**Objective:** This study aimed to examine publication patterns, identify core keywords and the underlying knowledge structure, and characterize major research topics and their temporal trends in research on contactless wireless sensing for health monitoring of older adults.

**Methods:** We analyzed the abstracts of 328 journal articles retrieved from 5 databases up to October 30, 2026. A keyword co-occurrence network was constructed from keywords ranked within the top 60 by term frequency (TF), followed by analyses of network centrality and cluster structure. Research topics were identified using BERTopic, and simple linear regression was used to analyze annual topic proportions and classify temporal trends as increasing or decreasing.

**Results:** Among the 328 articles, 71.6% (235/328) were published in 2020 or later. The predominant sensing technology shifted from radio frequency identification (RFID; 58.8% in 2006–2014) to radar (64.7% in 2020 or later). The keyword network comprised 4 clusters, with the sleep, vital sign, and behavior monitoring cluster located at the periphery. Of the 7 topics, 3 belonged to the fall detection domain (99 articles, 30.2%). Deep learning–based radar fall detection showed a significant increasing trend (B=2.27, P<.001), whereas RFID-based indoor location and care management showed a significant decreasing trend (B=−4.10, P<.001); both trends remained consistent in the sensitivity analyses. Eight (2.4%) articles examined the detection of bed-related movements, such as bed exit, and 2 (0.6%) examined the detection of behavioral symptoms in older adults with dementia.

**Conclusions:** Research on contactless wireless sensing for health monitoring of older adults has expanded, with increasing emphasis on deep learning–based radar fall detection. However, studies focusing on bed-exit detection and the monitoring of behavioral symptoms in older adults with dementia remain limited. Interdisciplinary collaboration between engineering and health care researchers is needed to validate these technologies in the real-life and care settings for older adults.

## Files

- `included_studies.xlsx`: 328 included studies (DOI, PMID). Abstracts are not included because of database copyright.
- `synonyms.xlsx`, `special_words.xlsx`, `stopwords.xlsx`: dictionaries
- `word_statistics_2.xlsx`, `CoWord_62_w14.xlsx`, `Jaccard_62.xlsx`, `louvain.xlsx`: NetMiner results
- `bertopic_2_doc_topics.xlsx`: BERTopic topic assignment
- `01_preprocessing.py`, `02_topic_modeling.py`, `03_topic_trends.py`: analysis code
