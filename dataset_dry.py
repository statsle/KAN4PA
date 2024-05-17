import pandas as pd

class SignificantFiguresCounter:
    def __init__(self, site):
        self.site = site
        self.data = pd.read_excel(site, sheet_name='data')
        self.tot = 0
        self.tot_1 = 0
        self.significant_figures = []
        self.age = []

    def contains_digit(self, number, digit):
        number_str = str(number)
        digit_str = str(digit)
        return digit_str in number_str

    def count_significant_figures(self):
        for row in range(200):
            self.age.append(self.data.iloc[row, 10])
            
            flag = 1
            row_data = []
            for col in range(15):
                selected_value = self.data.iloc[row, col]
                row_data.append(selected_value)
                if pd.isna(selected_value):
                    flag = 0
            
            if flag == 1:        
                if row_data[1] == 1:
                    self.tot_1 += 1
                for i in range(1, 6):
                    onehot_ = 1.0 if self.contains_digit(number=row_data[10], digit=i) else 0.0
                    row_data.append(onehot_)
                for i in range(1, 6):
                    onehot_ = 1.0 if self.contains_digit(number=row_data[13], digit=i) else 0.0
                    row_data.append(onehot_)
                self.significant_figures.append(row_data)
            self.tot += flag
        
        print(f'total_significant_figures={self.tot}')
        print(f'label=1={self.tot_1}')
        
        return self.significant_figures, self.age
