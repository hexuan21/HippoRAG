import json
import os
from tqdm import tqdm
from hipporag.evaluation.qa_eval import QAExactMatch,QAF1Score
from utils import _llm_azure_api,_llm_openrouter_api,MAX_TRY
import numpy as np


def llm_acc_check(judge_model_name,res_path,use_azure,):
    res_list=json.load(open(res_path,"r"))
    judge_list=[]
    judge_path=res_path.replace(".json","_judge.json")
    
    correct_num=0
    total_num=len(res_list)
    for res_item in tqdm(res_list):
        id=res_item["id"]
        q=res_item["question"]
        ref=res_item["ref_answer"]
        ans=res_item["model_answer"]
        prompt=f"You are a helpful chat assistant. Please check and determine if the student's answer to the question is correct according to the reference answer. \nHere's the reference answer: {ref}. \nHere's the student's answer: {ans}. If correct, output '1'; else, output '0'. DO NOT include anything else before or after your answer."
        
        try_counts=0
        while True:
            if try_counts>=MAX_TRY:
                break
            if use_azure:
                res=_llm_azure_api(prompt,judge_model_name)
            else:
                res=_llm_openrouter_api(prompt,judge_model_name)
            try_counts+=1
            if len(res)>0:
                break
            print(f"trying again, {try_counts}-try (max:{MAX_TRY} times)")
            
        judge_list.append({
            "id":id,
            "question":q,
            "ref_answer":ref,
            "model_answer":ans,
            "judge":res,
        })
        if "1" in res:
            correct_num+=1
        
    print(f"Accuracy: {correct_num/total_num}({correct_num}/{total_num})")
        
    with open(judge_path,"w") as f:
        json.dump(judge_list,f,indent=4)


def eval(res_path):
    res_list=json.load(open(res_path,"r"))
    
    gold_answers,predicted_answers=[[x["ref_answer"]] for x in res_list],[x["model_answer"] for x in res_list]
    qa_em_evaluator = QAExactMatch()
    qa_f1_evaluator = QAF1Score()
    overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
        gold_answers=gold_answers, predicted_answers=predicted_answers,
        aggregation_fn=np.max)
    overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
        gold_answers=gold_answers, predicted_answers=predicted_answers,
        aggregation_fn=np.max)

    # round off to 4 decimal places for QA results
    overall_qa_em_result.update(overall_qa_f1_result)
    overall_qa_results = overall_qa_em_result
    overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
    print(f"Evaluation results for QA: {overall_qa_results}")    
    
    
if __name__ == "__main__":
    res_dir="baseline/res"
    os.makedirs(res_dir,exist_ok=True)
    
    res_path=f"{res_dir}/hotpotqa_gpt-4o.json"
    eval(res_path)
    
    # use_azure=0
    # judge_model_name="gpt-4o-mini"
    # res_path=f"{res_dir}/hotpotqa_gpt-4o.json"
    # llm_acc_check(udge_model_name,res_path,use_azure,j)