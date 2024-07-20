import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


language = None


def read_and_categorize_data(file_path, sheet_name):
    """
    Read data from an excel file and categorize it.
    :param file_path: Path to the file
    :param sheet_name: Name of the sheet to read
    :return:
    """
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Create an empty dictionary to store categorized data
    category_data = {}
    # Categorize by category
    categories = df['Category'].unique()
    for category in categories:
        category_data[category] = {}
        category_df = df[df['Category'] == category]
        groups = category_df['Group'].unique()
        for group in groups:
            group_df = category_df[category_df['Group'] == group]
            # Drop the category and group columns, keep other columns' data
            data_without_category_group = group_df.drop(columns=['Category', 'Group']).values.tolist()
            category_data[category][group] = data_without_category_group
    return category_data


def plot_results(category, group, x_train, y_train, x_test, y_test, y_pred, coef_intercept, r2):
    """
    Plot the fitting results.
    :param category: Type of category
    :param group: Group name
    :param x_train: Feature data for training
    :param y_train: Target data for training
    :param x_test: Feature data for testing
    :param y_test: Target data for testing
    :param y_pred: Predicted results
    :param coef_intercept: Slope and intercept of the fitting function
    :param r2: R^2 result
    :return: No return value
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei to display Chinese
    plt.rcParams['axes.unicode_minus'] = False  # Normal display of negative sign
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='blue', label='Training data')
    plt.scatter(x_test, y_test, color='purple', label='Testing data')
    plt.plot(x_test, y_pred, color='red', label='Linear fit line')
    plt.scatter(x_test, y_pred, color='green', label='Predicted data')

    # Add regression equation and variance to legend
    equation = f"y = {coef_intercept['coef']:.2f}x + {coef_intercept['intercept']:.2f}"
    variance = f"R^2 = {r2:.2f}"

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='w', markerfacecolor='black', marker='_', markersize=0, label=equation),
                    Line2D([0], [0], color='w', markerfacecolor='black', marker='_', markersize=0, label=variance)]

    if language == '中文':
        plt.legend(handles=[*plt.gca().get_legend_handles_labels()[0], *custom_lines],
                   labels=['训练集数据', '测试集数据', '线性拟合直线', '预测数据', equation, variance], loc='best')
        plt.title(f'【{category} - {group}】线性回归图')
    else:
        plt.legend(handles=[*plt.gca().get_legend_handles_labels()[0], *custom_lines],
                   labels=['Training data', 'Testing data', 'Linear fit line', 'Predicted data', equation, variance],
                   loc='best')
        plt.title(f'【{category} - {group}】Linear Regression Plot')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def k_fold(df, x_index, y_index, k_num=5, is_plot=False):
    """
    Perform linear fitting and calculate using k-fold cross-validation.
    :param df: Dataset
    :param x_index: Feature position to read
    :param y_index: Target position to read
    :param k_num: Number of folds for k-fold cross-validation, default is 5, data type: int
    :param is_plot: Whether to plot images, data type: bool
    :return: Various calculation results
    """
    kf = KFold(n_splits=k_num, shuffle=True, random_state=42)  # 5-fold cross-validation
    l_model = LinearRegression()  # Linear regression model
    mse_results = {}  # Store MSE results for each group
    coef_intercept_results = {}  # Store slopes and intercepts for each group
    r2_results = {}  # Store R^2 results for each group
    train_mse_results = {}  # Store training MSE results for each group
    test_mse_results = {}  # Store testing MSE results for each group
    for category, groups in df.items():
        if category not in mse_results:
            mse_results[category] = {}
            coef_intercept_results[category] = {}
            r2_results[category] = {}
            train_mse_results[category] = {}
            test_mse_results[category] = {}
        for group, data in groups.items():
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            x_data = df.iloc[:, x_index].values.reshape(-1, 1)
            y = df.iloc[:, y_index].values
            mse_scores = []  # MSE values
            coef_intercept_scores = []  # Slopes and intercepts
            r2_scores = []  # R^2 values
            train_mse_scores = []  # Training MSE values
            test_mse_scores = []  # Testing MSE values
            for train_index, test_index in kf.split(x_data):
                x_train, x_test = x_data[train_index], x_data[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # Train the model
                l_model.fit(x_train, y_train)
                # Predict
                y_train_pred = l_model.predict(x_train)
                y_test_pred = l_model.predict(x_test)
                # Calculate training and testing MSE
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_mse_scores.append(train_mse)
                test_mse_scores.append(test_mse)
                # Calculate MSE
                mse = mean_squared_error(y_test, y_test_pred)
                mse_scores.append(mse)
                # Calculate R^2
                r2 = r2_score(y_test, y_test_pred)
                r2_scores.append(r2)
                # Get slope and intercept
                coef_intercept = {'coef': l_model.coef_[0], 'intercept': l_model.intercept_}
                coef_intercept_scores.append(coef_intercept)
                # Plot results
                if is_plot:
                    plot_results(category, group, x_train, y_train, x_test, y_test, y_test_pred, coef_intercept, r2)
            # Store MSE results for each group
            mse_results[category][group] = mse_scores
            # Store slopes and intercepts for each group
            coef_intercept_results[category][group] = coef_intercept_scores
            # Store R^2 results for each group
            r2_results[category][group] = r2_scores
            # Store training MSE results for each group
            train_mse_results[category][group] = train_mse_scores
            # Store testing MSE results for each group
            test_mse_results[category][group] = test_mse_scores
    return mse_results, coef_intercept_results, r2_results, train_mse_results, test_mse_results


def print_mse_results(mse_results, coef_intercept_results, r2_results, train_mse_results, test_mse_results):
    """
    Parse data and display output in a friendly manner.
    :param mse_results: MSE calculation results
    :param coef_intercept_results: Slopes and intercepts calculation results
    :param r2_results: R^2 calculation results
    :param train_mse_results: Training MSE calculation results
    :param test_mse_results: Testing MSE calculation results
    :return:
    """
    if language == '中文':
        for category, groups in mse_results.items():
            print(f"类别: {category}")
            for group, mse_scores in groups.items():
                print(f"  分组: {group}")
                print(f"    均方差值（MSE）: {', '.join(map(str, mse_scores))}")
                print(f"    平均均方差值（MSE）: {sum(mse_scores) / len(mse_scores)}")
                # Output training and testing MSE values
                train_mse_scores = train_mse_results[category][group]
                test_mse_scores = test_mse_results[category][group]
                print(f"    训练集MSE值: {', '.join(map(str, train_mse_scores))}")
                print(f"    测试集MSE值: {', '.join(map(str, test_mse_scores))}")
                # Output R^2 values
                r2_scores = r2_results[category][group]
                print(f"    R^2值: {', '.join(map(str, r2_scores))}")
                print(f"    平均R^2值: {sum(r2_scores) / len(r2_scores)}")
                # Output slope and intercept
                coef_intercept_scores = coef_intercept_results[category][group]
                print(f"    斜率和截距:")
                for i, coef_intercept in enumerate(coef_intercept_scores):
                    print(f"      第 {i + 1} 折: 斜率 = {coef_intercept['coef']}, 截距 = {coef_intercept['intercept']}")
            print()
    else:
        for category, groups in mse_results.items():
            print(f"Category: {category}")
            for group, mse_scores in groups.items():
                print(f"  Group: {group}")
                print(f"    Mean Squared Error (MSE) values: {', '.join(map(str, mse_scores))}")
                print(f"    Average Mean Squared Error (MSE): {sum(mse_scores) / len(mse_scores)}")
                # Output training and testing MSE values
                train_mse_scores = train_mse_results[category][group]
                test_mse_scores = test_mse_results[category][group]
                print(f"    Training MSE values: {', '.join(map(str, train_mse_scores))}")
                print(f"    Testing MSE values: {', '.join(map(str, test_mse_scores))}")
                # Output R^2 values
                r2_scores = r2_results[category][group]
                print(f"    R^2 values: {', '.join(map(str, r2_scores))}")
                print(f"    Average R^2: {sum(r2_scores) / len(r2_scores)}")
                # Output slope and intercept
                coef_intercept_scores = coef_intercept_results[category][group]
                print(f"    Slopes and Intercepts:")
                for i, coef_intercept in enumerate(coef_intercept_scores):
                    print(
                        f"      Fold {i + 1}: Slope = {coef_intercept['coef']}, Intercept = {coef_intercept['intercept']}")
            print()


def colored_print(text, color_code):
    """
    Colored output.
    :param text: Text to output
    :param color_code: Color code
    :return: No return value
    """
    print(f"\033[{color_code}m{text}\033[0m")


def title_print(type):
    """
    Print the title of results.
    :param type: Current workbook name, data type: String
    :return: No return value
    """
    colored_print("=========================================================================", "1;31")
    if language == '中文':
        colored_print(f"||               🔖当前读取数据工作簿：{type}🔖               ||", "1;31")
    else:
        colored_print(f"||               🔖Current workbook being read: {type}🔖               ||", "1;31")
    colored_print("=========================================================================", "1;31")


def summary_cal(df, type, is_plot):
    """
    Coordinate linear fitting calculations and output related content.
    :param df: Dataset, data type DataFrame
    :param type: Current workbook name, data type String
    :param is_plot: Whether to plot images, data type Bool
    :return: Return -1 if error occurs
    """
    if type == "LDH等温":

        x_index, y_index = 2, 3
        title_print(type)
        mse_results, coef_intercept_results, r2_results, train_mse_results, test_mse_results = \
            k_fold(df, x_index, y_index, is_plot=is_plot)
        print_mse_results(mse_results, coef_intercept_results, r2_results, train_mse_results, test_mse_results)

        x_index, y_index = 0, 1
        title_print(type)
        mse_results, coef_intercept_results, r2_results, train_mse_results, test_mse_results = \
            k_fold(df, x_index, y_index, is_plot=is_plot)
        print_mse_results(mse_results, coef_intercept_results, r2_results, train_mse_results, test_mse_results)
    elif type == "LDH动力":

        x_index, y_index = 0, 3
        title_print(type)
        mse_results, coef_intercept_results, r2_results, train_mse_results, test_mse_results = \
            k_fold(df, x_index, y_index, is_plot=is_plot)
        print_mse_results(mse_results, coef_intercept_results, r2_results, train_mse_results, test_mse_results)

        x_index, y_index = 0, 4
        title_print(type)
        mse_results, coef_intercept_results, r2_results, train_mse_results, test_mse_results = \
            k_fold(df, x_index, y_index, is_plot=is_plot)
        print_mse_results(mse_results, coef_intercept_results, r2_results, train_mse_results, test_mse_results)
    else:
        if language == '中文':
            print("错误：错误的工作簿名称，请检查!")
        else:
            print("Error: Invalid workbook name, please check!")
        return -1


def main():
    _GREEN_ = '\033[92m'
    _RESET_ = '\033[0m'

    filepath = "DataSet_new.xlsx"  # File path, default in current path
    sheet_name = None  # Select file workbook
    is_plot = None  # Whether to plot regression line graph

    # Language selection
    language = input(_GREEN_ + "Please select language（English/中文）: " + _RESET_)
    while language not in ['中文', 'English']:
        print(_GREEN_ + "The input is invalid, please enter 中文 or English." + _RESET_)
        language = input(_GREEN_ + "Please select language（English/中文）: " + _RESET_)

    # Read workbook
    if language == 'English':
        sheet_name = input(_GREEN_ + "Please select the workbook to read 1.LDH isothermal 2.LDH dynamic (1/2): " + _RESET_)
        while sheet_name not in ['1', '2']:
            print(_GREEN_ + "Invalid input, please enter 1 or 2" + _RESET_)
            sheet_name = input(_GREEN_ + "Please select the workbook to read 1.LDH isothermal 2.LDH dynamic (1/2): " + _RESET_)
    else:
        sheet_name = input(_GREEN_ + "请选择读取工作簿 1.LDH等温 2.LDH动力（1/2）: " + _RESET_)
        while sheet_name not in ['1', '2']:
            print(_GREEN_ + "输入无效, 请输入 1 或者 2" + _RESET_)
            sheet_name = input(_GREEN_ + "请选择读取工作簿 1.LDH等温 2.LDH动力（1/2）: " + _RESET_)

    # Plot fitting result
    if language == 'English':
        is_plot = input(_GREEN_ + "Do you want to plot the fitting result for each fold (T/F): " + _RESET_)
        while is_plot not in ['T', 'F']:
            print(_GREEN_ + "Invalid input, please enter T or F" + _RESET_)
            is_plot = input(_GREEN_ + "Do you want to plot the fitting result for each fold (T/F): " + _RESET_)
    else:
        is_plot = input(_GREEN_ + "是否绘制每一折拟合结果图 (T/F): " + _RESET_)
        while is_plot not in ['T', 'F']:
            print(_GREEN_ + "输入无效, 请输入 T 或者 F" + _RESET_)
            is_plot = input(_GREEN_ + "是否绘制每一折拟合结果图 (T/F): " + _RESET_)

    sheet_name_mapping = {
        "1": "LDH等温",
        "2": "LDH动力"
    }
    sheet_name = sheet_name_mapping.get(sheet_name, "LDH等温")    # Verify reading file
    is_plot = is_plot == "T"    # Verify whether to plot images
    data_set = read_and_categorize_data(filepath, sheet_name)   # Read data
    summary_cal(data_set, sheet_name, is_plot)      # Fit model and cross-validate


if __name__ == '__main__':
    main()