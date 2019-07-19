# importing the config file, here inside /bayronds/challenges/pdf_parsing
import json

with open('../../../bayronds_config.json', 'r') as f:
    config = json.load(f)

# setting the dataset directory
dataset_directory = config['DEFAULT']['DATASET_DIRECTORY']
tika_directory = config['DEFAULT']['TIKA_DIRECTORY']
private_directory = config['DEFAULT']['PRIVATE_DIRECTORY']

#######################################

import numpy as np

class FileLoader:
    def __init__(self):
        self.skill2id = {}
        self.skills = {}
        self.subskills_ftvec = {}
        self.query2id = {}
        self.queries = {}
        self.querytokens_ftvec = {}
        self.expert2id = {}
        self.experts = {}
        self.document2expert = {}
        self.documents = {}

    def _load_skills(self):
        with open(private_directory + "/bayron/dicts/skill2id.json", "r") as f:
            skill2id = json.load(f)
        with open(private_directory + "/bayron/dicts/skills.json", "r") as f:
            raw_skills = json.load(f)
        skills = {}
        for key, skill in raw_skills.items():
            skills[int(key)] = skill
            skills[int(key)]["skill_docvec"] = np.array(json.loads(skill["skill_docvec"]))
            skills[int(key)]["skill_ft_meanvec"] = np.array(json.loads(skill["skill_ft_meanvec"]))  
            skills[int(key)]["skill_bp_docvec"] = np.array(json.loads(skill["skill_bp_docvec"]))    
            skills[int(key)]["skill_bpft_docvec"] = np.array(json.loads(skill["skill_bpft_docvec"]))    
        with open(private_directory + "/bayron/dicts/subskills_ftvec.json", "r") as f:
            subskills_ftvec = json.load(f)
        for key, value in subskills_ftvec.items():
            value["vector"] = np.array(value["vector"])

        self.skill2id = skill2id
        self.skills = skills
        self.subskills_ftvec = subskills_ftvec

    def _load_queries(self):
        with open(private_directory + "/bayron/dicts/query2id.json", "r") as f:
            query2id = json.load(f)
        with open(private_directory + "/bayron/dicts/queries.json", "r") as f:
            raw_queries = json.load(f)
        queries = {}
        for key, query in raw_queries.items():
            queries[int(key)] = query
            queries[int(key)]["query_docvec"] = np.array(json.loads(query["query_docvec"]))
            queries[int(key)]["query_noStopwords_docvec"] = np.array(json.loads(query["query_noStopwords_docvec"]))

        with open(private_directory + "/bayron/dicts/querytokens_ftvec.json", "r") as f:
            querytokens_ftvec = json.load(f)
        for key, value in querytokens_ftvec.items():
            value["vector"] = np.array(value["vector"])
        
        self.query2id = query2id
        self.queries = queries
        self.querytokens_ftvec = querytokens_ftvec
        

    def _load_experts(self):
        with open(private_directory + "/bayron/dicts/expert2id.json", "r") as f:
            expert2id = json.load(f)
        with open(private_directory + "/bayron/dicts/experts.json", "r") as f:
            raw_experts = json.load(f)
        experts = {}
        for key, expert in raw_experts.items():
            experts[int(key)] = expert
        
        self.expert2id = expert2id
        self.experts = experts
        

    def _load_documents(self):
        with open(private_directory + "/bayron/bayron_docs.txt", "rb") as f:
            bags = json.load(f)
        
        # turn bags dictionary into documents dictionary
        documents = {}
        document2expert = {}
        did = 0
        for expert, expert_dict in bags.items():
            if expert not in self.expert2id.keys():
                pass
            else:
                for file_name, document in expert_dict.items():
                    documents[did] = document
                    document2expert[did] = expert
                    did += 1

        self.document2expert = document2expert
        self.documents = documents

    def load(self):
        self._load_skills()
        self._load_queries()
        self._load_experts()
        self._load_documents()

f = FileLoader()
f.load()

def load_data(private_directory, with_documents=True):
    with open(private_directory + "/bayron/dicts/skill2id.json", "r") as f:
        skill2id = json.load(f)
    with open(private_directory + "/bayron/dicts/skills.json", "r") as f:
        raw_skills = json.load(f)
    skills = {}
    for key, skill in raw_skills.items():
        skills[int(key)] = skill
        skills[int(key)]["skill_docvec"] = np.array(json.loads(skill["skill_docvec"]))
        skills[int(key)]["skill_ft_meanvec"] = np.array(json.loads(skill["skill_ft_meanvec"]))
        skills[int(key)]["skill_bpft_docvec"] = np.array(json.loads(skill["skill_bpft_docvec"]))
        skills[int(key)]["skill_bp_docvec"] = np.array(json.loads(skill["skill_bp_docvec"]))
        
    # delete object to save ram
    del raw_skills

    with open(private_directory + "/bayron/dicts/subskills_ftvec.json", "r") as f:
        subskills_ftvec = json.load(f)
    for key, value in subskills_ftvec.items():
        value["vector"] = np.array(value["vector"])
        
    with open(private_directory + "/bayron/dicts/query2id.json", "r") as f:
        query2id = json.load(f)
    with open(private_directory + "/bayron/dicts/queries.json", "r") as f:
        raw_queries = json.load(f)
    queries = {}
    for key, query in raw_queries.items():
        queries[int(key)] = query
        queries[int(key)]["query_docvec"] = np.array(json.loads(query["query_docvec"]))
        queries[int(key)]["query_noStopwords_docvec"] = np.array(json.loads(query["query_noStopwords_docvec"]))

    # delete object to save ram
    del raw_queries

    with open(private_directory + "/bayron/dicts/querytokens_ftvec.json", "r") as f:
        querytokens_ftvec = json.load(f)
    for key, value in querytokens_ftvec.items():
        value["vector"] = np.array(value["vector"])
        
    with open(private_directory + "/bayron/dicts/expert2id.json", "r") as f:
        expert2id = json.load(f)
    with open(private_directory + "/bayron/dicts/experts.json", "r") as f:
        raw_experts = json.load(f)
    experts = {}
    for key, expert in raw_experts.items():
        experts[int(key)] = expert
        
    # delete object to save ram
    del raw_experts

    data = {"queries": queries, "experts": experts, "skills":skills, "query2id":query2id, "expert2id":expert2id, "skill2id":skill2id}

    if with_documents:
        with open(private_directory + "/bayron/bayron_docs.txt", "rb") as f:
            bags = json.load(f)
        
        # turn bags dictionary into documents dictionary
        documents = {}
        document2expert = {}
        did = 0
        for expert, expert_dict in bags.items():
            if expert not in expert2id.keys():
                pass
            else:
                for file_name, document in expert_dict.items():
                    documents[did] = document
                    document2expert[did] = expert
                    did += 1

        return data, documents, document2expert
    else:
        return data