# FKIE_itf_Task1

This is the repository for the paper "Using Small Densely Fully Connected Neural Nets for Event Detection and Clusering" at Socio-political and Crisis Events Detection at CASE @ ACL-IJCNLP 2021.
The Github of the workshop can be found here: https://github.com/emerging-welfare/case-2021-shared-task.
More information about the workshop are given here: https://emw.ku.edu.tr/case-2021/.
In each directory of this repository, you find a readme file with more instructions to use this work.


## Subtask 1
The first subtask is about detecting events about riots and social movements in news articles. A single training instance looks like this:

> {"id":100010,"text":"Smaller parties demand more time in parliament 08th March 2012 01:49 AM NEW DELHI: A day after the regional parties trounced the Congress in four states Assembly elections, the smaller parties flexed their muscles at a meeting of the chief whips demanding more time from the beleaguered government to raise various issues in both houses of the Parliament. On the eve of the Budget session, Parliamentary Affairs Minister Pawan Kumar Bansal convened the meeting of chief whips of various political parties to seek th","label":0}



## Subtask 2
The second subtask is about detecting events about riots and social movements in sentences. It's quite similar to the first task.
A train instance may look like this:



## Subtask 3

The third subtask is to identify which sentences are about the same event. A train instance looks like this:


>{"event_clusters":[[4,5],[3]],"sentence_no":[3,4,5],"sentences":["The Tamil Nadu Toddy Movement , which comprises about 300 farmer ’ s associations in the State , launched a law breaking protest on January 21 , 2009 demanding the ban on toddy lifted.",
"On the same day TNCC President Thangkabalu led a fast in Chennai in favor of total prohibition including toddy .",
"The counter protest by the TNCC chief raised a furore with the Movement with its coordinator S Nallusamy saying Thangkabalu “ had insulted the feelings of farmers , coconut and Palmyra tree climbers consumers and nature lovers . ” A few days back Thangkabalu told reporters in Salem that he had decided on the date for his protest three months ago and did not intend it to be a counter protest to that started by the toddy farmers ."],"id":55194}








## Requirements

- Python 3.8
- Flair 0.7
- networkx 2.5
- Keras 2.4.3
- numpy 1.19.2
- scikit-learn 0.24.0
- matplotlib 3.3.3

If you have any questions or problems, don't hesitate to open an issue or contact me at: nils.becker@fkie.fraunhofer.de