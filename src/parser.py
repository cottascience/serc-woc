import numpy as np
from collections import defaultdict, OrderedDict
import sys, pickle, csv, torch
from itertools import product

intensity = {"Never": 1.0, "Rarely": 2.0, "Sometimes": 3.0, "Very often": 4.0, "Always": 5.0}
freedom = {"No freedom at all": 1.0, "Little freedom": 2.0, "Moderate freedom": 3.0, "Much freedom": 4.0, "Complete freedom": 5.0}
clarity = {"Unclear": 1.0, "Somewhat unclear": 2.0, "Neither clear nor unclear": 3.0, "Somewhat clear": 4.0, "Clear": 5.0}
certainty = {"Definitely not": 1.0, "Probably not": 2.0, "Might or might not": 3.0, "Probably yes": 4.0, "Definitely yes": 5.0}
availability = {"Very low availability": 1.0, "Low availability": 2.0, "Moderate availability": 3.0, "High availability": 4.0, "Very high availability": 5.0}
increase = {"Large decrease": 1.0, "Slight decrease": 2.0, "No change": 3.0, "Slight increase": 4.0, "Large increase": 5.0}
schedule = {"On schedule": 0.0, "Behind schedule": 1.0, "Ahead of schedule": 1.0,}
budget = {"On budget": 0.0, "Under budget": 1.0, "Over budget": 1.0,}
objs = {"Meeting requirements as planned": 0.0, "Meeting fewer requirements than planned": 1.0, "Meeting more requirements than planned": 1.0,}
RR = {"0": 0.0, "1": 1.0}

def one_hot_encoding(string, string_levels):
    string_levels = sorted(list(string_levels))
    encoding = [0.0]*len(string_levels)
    encoding[string_levels.index(string)] = 1.0
    return encoding

class Data:
    def __init__(self, path, device, y_lab):
        self.student = self.load_csv(path+"student.csv")
        self.instructor = self.load_csv(path+"instructor.csv")
        self.device = device
        self.x_ids = None
        self.x = []
        self.y = []
        self.prev_y = []
        self.val_x_ids = None
        self.val_x = []
        self.val_y = []
        self.val_prev_y = []
        self.test_x_ids = None
        self.test_x = []
        self.test_y = []
        self.test_prev_y = []
        self.x_sizes = OrderedDict()
        self.num_features = 0
        self.num_classes = len(set(schedule.values()))
        self.all_ids = []
        self.y_lab = y_lab
        if y_lab == "PROJS":
            self.y_dict = budget
        elif y_lab == "PROJT":
            self.y_dict = schedule
        elif y_lab == "PROJP":
            self.y_dict = objs
        else:
            self.y_dict = RR
        project_week = defaultdict(list)
        for i in range(len(self.instructor)):
            project_week[self.instructor[i]["Project"]].append(int(self.instructor[i]["Week"]))
        for i in range(len(self.student)):
            if (int(self.student[i]["Week"]) + 1) in project_week[self.student[i]["Project"]]:
                self.all_ids.append((int(self.student[i]["Week"]),self.student[i]["Project"]))
        self.all_ids = list(set(self.all_ids))
        self.num_examples = len(self.all_ids)
    def load_csv(self, file_path):
        with open(file_path) as f:
            reader = csv.DictReader(f, delimiter=';')
            next(reader)
            data = []
            for row in reader:
                data.append(row)
        return data
    def column(self, data, c):
        return [data[i][c] for i in range(len(data))]
    def reset(self):
        self.x_ids = None
        self.x = []
        self.y = []
        self.prev_y = []
        self.val_x_ids = None
        self.val_x = []
        self.val_y = []
        self.val_prev_y = []
        self.test_x_ids = None
        self.test_x = []
        self.test_y = []
        self.test_prev_y = []
    def generate_dataset(self, ids, split="train"):
        id_mapping = []
        for ex, project_week in enumerate(self.all_ids):
            label_exists = False
            prev_label_exists = False
            if ex in ids:
                for i in range(len(self.instructor)):
                    if int(self.instructor[i]["Week"]) == (project_week[0] + 1) and self.instructor[i]["Project"] == project_week[1]:
                        if split == "train":
                            self.y.append(self.y_dict[self.instructor[i][self.y_lab]])
                        elif split == "test":
                            self.test_y.append(self.y_dict[self.instructor[i][self.y_lab]])
                        else:
                            self.val_y.append(self.y_dict[self.instructor[i][self.y_lab]])
                        label_exists = True
                if label_exists:
                    for i in range(len(self.instructor)):
                        if int(self.instructor[i]["Week"]) == (project_week[0]) and self.instructor[i]["Project"] == project_week[1]:
                            if split == "train":
                                self.prev_y.append(self.y_dict[self.instructor[i][self.y_lab]])
                            elif split == "test":
                                self.test_prev_y.append(self.y_dict[self.instructor[i][self.y_lab]])
                            else:
                                self.val_prev_y.append(self.y_dict[self.instructor[i][self.y_lab]])
                            prev_label_exists = True
                if len(self.y) > len(self.prev_y): self.y = self.y[:-1]
                if len(self.val_y) > len(self.val_prev_y): self.val_y = self.val_y[:-1]
                if len(self.test_y) > len(self.test_prev_y): self.test_y = self.test_y[:-1]
                if label_exists and prev_label_exists:
                    usr_count = 0
                    for i in range(len(self.student)):
                        features = []
                        if int(self.student[i]["Week"]) == project_week[0] and self.student[i]["Project"] == project_week[1]:
                            features.append(float(self.student[i]["EXP"]))
                            self.x_sizes["EXP"] = 1
                            features.append(float(self.student[i]["PRO"]))
                            self.x_sizes["PRO"] = 1
                            features.append(intensity[self.student[i]["SL"]])
                            self.x_sizes["SL"] = 1
                            features.append(intensity[self.student[i]["COO1"]])
                            self.x_sizes["COO1"] = 1
                            features.append(intensity[self.student[i]["IMP"]])
                            self.x_sizes["IMP"] = 1
                            features.append(float(self.student[i]["COO2"]))
                            self.x_sizes["COO2"] = 1
                            features.append(freedom[self.student[i]["STND"]])
                            self.x_sizes["STND"] = 1
                            features.append(float(self.student[i]["AUTO"]))
                            self.x_sizes["AUTO"] = 1
                            #MODU = float(self.student[i]["MODU"])
                            features.append(clarity[self.student[i]["COBJ"]])
                            self.x_sizes["COBJ"] = 1
                            features.append(certainty[self.student[i]["COMT"]])
                            self.x_sizes["COMT"] = 1
                            features.append(availability[self.student[i]["RESO"]])
                            self.x_sizes["RESO"] = 1
                            features.append(intensity[self.student[i]["COMM"]])
                            self.x_sizes["COMM"] = 1
                            features.append(intensity[self.student[i]["NEUR"]])
                            self.x_sizes["NEUR"] = 1
                            features.append(intensity[self.student[i]["OPEN"]])
                            self.x_sizes["OPEN"] = 1
                            features.append(intensity[self.student[i]["CONS"]])
                            self.x_sizes["CONS"] = 1
                            features.append(intensity[self.student[i]["EXTR"]])
                            self.x_sizes["EXTR"] = 1
                            features.append(intensity[self.student[i]["AGREE"]])
                            self.x_sizes["AGREE"] = 1
                            features.append(one_hot_encoding(self.student[i]["PROJS"], set(self.column(self.student, "PROJS"))))
                            self.x_sizes["PROJS"] = len(set(self.column(self.student, "PROJS")))
                            features.append(one_hot_encoding(self.student[i]["PROJT"], set(self.column(self.student, "PROJT"))))
                            self.x_sizes["PROJT"] = len(set(self.column(self.student, "PROJT")))
                            features.append(one_hot_encoding(self.student[i]["PROJP"], set(self.column(self.student, "PROJP"))))
                            self.x_sizes["PROJP"] = len(set(self.column(self.student, "PROJP")))
                            features.append(float(self.student[i]["PROJSC"]))
                            self.x_sizes["PROJSC"] = 1
                            features.append(float(self.student[i]["PROJTC"]))
                            self.x_sizes["PROJTC"] = 1
                            features.append(float(self.student[i]["PROJPC"]))
                            self.x_sizes["PROJPC"] = 1
                            features.append(one_hot_encoding(self.student[i]["UNEFF"], set(self.column(self.student, "UNEFF"))))
                            self.x_sizes["UNEFF"] = len(set(self.column(self.student, "UNEFF")))
                            features.append(one_hot_encoding(self.student[i]["COBJ"], set(self.column(self.student, "COBJ"))))
                            self.x_sizes["COBJ"] = len(set(self.column(self.student, "COBJ")))
                            features.append(one_hot_encoding(self.student[i]["FSYM"], set(self.column(self.student, "FSYM"))))
                            self.x_sizes["FSYM"] = len(set(self.column(self.student, "FSYM")))
                            features.append(one_hot_encoding(self.student[i]["BUREAU"], set(self.column(self.student, "BUREAU"))))
                            self.x_sizes["BUREAU"] = len(set(self.column(self.student, "BUREAU")))
                            features.append(increase[self.student[i]["OUTP"]])
                            self.x_sizes["OUTP"] = 1
                            features.append(one_hot_encoding(self.student[i]["SMENG"], set(self.column(self.student, "SMENG"))))
                            self.x_sizes["SMENG"] = self.x_sizes["SMENG"] = len(set(self.column(self.student, "SMENG")))
                            features.append(one_hot_encoding(self.student[i]["EAT1"], set(self.column(self.student, "EAT1"))))
                            self.x_sizes["EAT1"] = self.x_sizes["EAT1"] = len(set(self.column(self.student, "EAT1")))
                            features.append(float(self.student[i]["TSPENT"]))
                            self.x_sizes["TSPENT"] = 1
                            features.append(float(self.student[i]["TMEET"]))
                            self.x_sizes["TMEET"] = 1
                            features.append(float(self.student[i]["NTOOL"]))
                            self.x_sizes["NTOOL"] = 1
                            features.append(one_hot_encoding(self.student[i]["EXERC"], set(self.column(self.student, "EXERC"))))
                            self.x_sizes["EXERC"] = len(set(self.column(self.student, "EAT1")))
                            features.append(float(self.student[i]["FPRES"]))
                            self.x_sizes["FPRES"] = 1
                            features.append(one_hot_encoding(self.student[i]["AMBI"], set(self.column(self.student, "AMBI"))))
                            self.x_sizes["AMBI"] = len(set(self.column(self.student, "AMBI")))
                            features.append(one_hot_encoding(self.student[i]["BANDW"], set(self.column(self.student, "BANDW"))))
                            self.x_sizes["BANDW"] = len(set(self.column(self.student, "BANDW")))
                            features.append(one_hot_encoding(self.student[i]["FOCUS"], set(self.column(self.student, "FOCUS"))))
                            self.x_sizes["FOCUS"] = len(set(self.column(self.student, "FOCUS")))
                            features.append(one_hot_encoding(self.student[i]["NORM"], set(self.column(self.student, "NORM"))))
                            self.x_sizes["NORM"] = len(set(self.column(self.student, "NORM")))
                            features.append(one_hot_encoding(self.student[i]["NOTIH"], set(self.column(self.student, "NOTIH"))))
                            self.x_sizes["NOTIH"] = len(set(self.column(self.student, "NOTIH")))
                            features.append(one_hot_encoding(self.student[i]["CONF"], set(self.column(self.student, "CONF"))))
                            self.x_sizes["CONF"] = len(set(self.column(self.student, "CONF")))
                            features.append(one_hot_encoding(self.student[i]["PARKL"], set(self.column(self.student, "PARKL"))))
                            self.x_sizes["PARKL"] = len(set(self.column(self.student, "PARKL")))
                            features.append(one_hot_encoding(self.student[i]["ANCHOR"], set(self.column(self.student, "ANCHOR"))))
                            self.x_sizes["ANCHOR"] = len(set(self.column(self.student, "ANCHOR")))
                            features.append(float(self.student[i]["OVERC"]))
                            self.x_sizes["OVERC"] = 1
                            for idx, item in enumerate(features):
                                if isinstance(item, float):
                                    features[idx] = [item]
                            features = sum(features,[])
                            if split == "train":
                                self.x.append(features)
                            elif split == "test":
                                self.test_x.append(features)
                            else:
                                self.val_x.append(features)
                            usr_count += 1
                    if usr_count > 0:
                        id_mapping.append((project_week,usr_count))
        if split == "train":
            self.x_ids = id_mapping
            self.y = torch.Tensor(self.y).long().to(self.device)
            self.prev_y = torch.Tensor(self.prev_y).long().to(self.device)
            self.x = torch.Tensor(self.x).float().to(self.device)
            self.num_features = self.x.size(1)
        elif split == "test":
            self.test_x_ids = id_mapping
            self.test_y = torch.Tensor(self.test_y).long().to(self.device)
            self.test_prev_y = torch.Tensor(self.test_prev_y).long().to(self.device)
            self.test_x = torch.Tensor(self.test_x).float().to(self.device)
            self.num_features = self.test_x.size(1)
        else:
            self.val_x_ids = id_mapping
            self.val_y = torch.Tensor(self.val_y).long().to(self.device)
            self.val_prev_y = torch.Tensor(self.val_prev_y).long().to(self.device)
            self.val_x = torch.Tensor(self.val_x).float().to(self.device)
            self.num_features = self.val_x.size(1)
