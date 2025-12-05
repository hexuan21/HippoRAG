python baseline/vanilla.py --use_azure --context_type "ten_psg" --model_name "gpt-4o"
python baseline/vanilla.py --use_azure --context_type "no_context" --model_name "gpt-4o"


python baseline/vanilla.py --context_type "ten_psg" --model_name "gpt-4o-mini"
python baseline/vanilla.py --context_type "no_context" --model_name "gpt-4o-mini"
python baseline/vanilla.py --context_type "all_corpus" --model_name "gpt-4o-mini"


python baseline/vanilla.py --context_type "ten_psg" --model_name "gpt-5-mini"
python baseline/vanilla.py --context_type "no_context" --model_name "gpt-5-mini"