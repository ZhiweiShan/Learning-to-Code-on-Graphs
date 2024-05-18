from fractions import Fraction
def sort_dict(data):
    fractions_dict = {}
    for key in data.keys():
        nums = list(map(int, key.split('-')))
        fraction = Fraction(nums[0], nums[1])
        fractions_dict[fraction] = data[key]

    # 按照分数的值从大到小排序
    sorted_fractions = sorted(fractions_dict.items(), reverse=True)

    # 打印排序后的分数及对应值的长度

    for fraction, value in sorted_fractions:
        print(f"{fraction}: {len(value)}")