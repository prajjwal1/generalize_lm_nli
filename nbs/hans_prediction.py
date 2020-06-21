import sys
from tqdm import tqdm

def format_label(label):
    if label == "entailment":
        return "entailment"
    else:
        return "non-entailment"
    
def get_predictions(file_path):
    fi = open(file_path, "r")
    first = True
    guess_dict = {}
    for line in fi:
        if first:
            first = False
            continue
        else:
            parts = line.strip().split(",")
            guess_dict[parts[0]] = format_label(parts[1])
    return guess_dict

def process_actual_preds(file_path):
    fi = open(file_path, "r")
    correct_dict = {}
    first = True

    heuristic_list = []
    subcase_list = []
    template_list = []

    for line in fi:
        if first:
            labels = line.strip().split("\t")
            idIndex = labels.index("pairID")
            first = False
            continue
        else:
            parts = line.strip().split("\t")
            this_line_dict = {}
            for index, label in enumerate(labels):
                if label == "pairID":
                    continue
                else:
                    this_line_dict[label] = parts[index]
            correct_dict[parts[idIndex]] = this_line_dict

            if this_line_dict["heuristic"] not in heuristic_list:
                heuristic_list.append(this_line_dict["heuristic"])
            if this_line_dict["subcase"] not in subcase_list:
                subcase_list.append(this_line_dict["subcase"])
            if this_line_dict["template"] not in template_list:
                template_list.append(this_line_dict["template"])
    return correct_dict, heuristic_list, subcase_list, template_list

def init_dicts(heuristic_list, subcase_list, template_list):
    heuristic_ent_correct_count_dict = {}
    subcase_correct_count_dict = {}
    template_correct_count_dict = {}
    heuristic_ent_incorrect_count_dict = {}
    subcase_incorrect_count_dict = {}
    template_incorrect_count_dict = {}
    heuristic_nonent_correct_count_dict = {}
    heuristic_nonent_incorrect_count_dict = {}



    for heuristic in heuristic_list:
        heuristic_ent_correct_count_dict[heuristic] = 0
        heuristic_ent_incorrect_count_dict[heuristic] = 0
        heuristic_nonent_correct_count_dict[heuristic] = 0 
        heuristic_nonent_incorrect_count_dict[heuristic] = 0

    for subcase in subcase_list:
        subcase_correct_count_dict[subcase] = 0
        subcase_incorrect_count_dict[subcase] = 0

    for template in template_list:
        template_correct_count_dict[template] = 0
        template_incorrect_count_dict[template] = 0
        
    return heuristic_ent_correct_count_dict, subcase_correct_count_dict, \
           template_correct_count_dict, heuristic_ent_incorrect_count_dict, \
           subcase_incorrect_count_dict, template_incorrect_count_dict, \
           heuristic_nonent_correct_count_dict, heuristic_nonent_incorrect_count_dict

def process_preds(heuristic_ent_correct_count_dict,
                 heuristic_nonent_correct_count_dict,
                 subcase_correct_count_dict,
                 template_correct_count_dict,
                 heuristic_ent_incorrect_count_dict,
                 heuristic_nonent_incorrect_count_dict,
                 subcase_incorrect_count_dict,
                template_incorrect_count_dict, correct_dict, guess_dict):
    for key in correct_dict:
        traits = correct_dict[key]
        heur = traits["heuristic"]
        subcase = traits["subcase"]
        template = traits["template"]

        guess = guess_dict[key]
        correct = traits["gold_label"]

        if guess == correct:
            if correct == "entailment":
                heuristic_ent_correct_count_dict[heur] += 1
            else:
                heuristic_nonent_correct_count_dict[heur] += 1

            subcase_correct_count_dict[subcase] += 1
            template_correct_count_dict[template] += 1
        else:
            if correct == "entailment":
                heuristic_ent_incorrect_count_dict[heur] += 1
            else:
                heuristic_nonent_incorrect_count_dict[heur] += 1
            subcase_incorrect_count_dict[subcase] += 1
            template_incorrect_count_dict[template] += 1
            
    return heuristic_ent_correct_count_dict,\
                 heuristic_nonent_correct_count_dict,\
                 subcase_correct_count_dict,\
                 template_correct_count_dict,\
                 heuristic_ent_incorrect_count_dict,\
                 heuristic_nonent_incorrect_count_dict,\
                 subcase_incorrect_count_dict,\
                template_incorrect_count_dict

def get_hans_results(heuristic_ent_correct_count_dict,
                    heuristic_nonent_correct_count_dict, heuristic_ent_incorrect_count_dict,
                     heuristic_nonent_incorrect_count_dict, heuristic_list):
    ent, non_ent = {}, {}
    for heuristic in heuristic_list:
        correct = heuristic_ent_correct_count_dict[heuristic]
        incorrect = heuristic_ent_incorrect_count_dict[heuristic]
        total = correct + incorrect
        percent_ent = correct * 1.0 / total
        ent[heuristic] = percent_ent
        
    for heuristic in heuristic_list:
        correct = heuristic_nonent_correct_count_dict[heuristic]
        incorrect = heuristic_nonent_incorrect_count_dict[heuristic]
        total = correct + incorrect
        percent_non_ent = correct * 1.0 / total
        non_ent[heuristic] = percent_non_ent
        
    return ent, non_ent

def get_hans_preds_main(hans_evaluation_set_file_path, hans_prediction_file_path):
    hans_main_ent = {}
    hans_main_non_ent = {}
    for file in tqdm(hans_prediction_file_path):
        guess_dict = get_predictions(file)
        correct_dict, heuristic_list, subcase_list, template_list = process_actual_preds(hans_evaluation_set_file_path )
        heuristic_ent_correct_count_dict, subcase_correct_count_dict, \
        template_correct_count_dict, heuristic_ent_incorrect_count_dict, \
        subcase_incorrect_count_dict, template_incorrect_count_dict, \
        heuristic_nonent_correct_count_dict, heuristic_nonent_incorrect_count_dict = init_dicts(heuristic_list,
                                                                                               subcase_list,
                                                                                               template_list,
                                                                                               )

        heuristic_ent_correct_count_dict,\
        heuristic_nonent_correct_count_dict,\
        subcase_correct_count_dict,\
        template_correct_count_dict,\
        heuristic_ent_incorrect_count_dict,\
        heuristic_nonent_incorrect_count_dict,\
        subcase_incorrect_count_dict,\
        template_incorrect_count_dict = process_preds(heuristic_ent_correct_count_dict,
                     heuristic_nonent_correct_count_dict,
                     subcase_correct_count_dict,
                     template_correct_count_dict,
                     heuristic_ent_incorrect_count_dict,
                     heuristic_nonent_incorrect_count_dict,
                     subcase_incorrect_count_dict,
                    template_incorrect_count_dict, correct_dict, guess_dict)

        ent, non_ent = get_hans_results(heuristic_ent_correct_count_dict,
                        heuristic_nonent_correct_count_dict, heuristic_ent_incorrect_count_dict,
                         heuristic_nonent_incorrect_count_dict, heuristic_list)
        hans_main_ent[file] = ent
        hans_main_non_ent[file] = non_ent
    return hans_main_ent, hans_main_non_ent