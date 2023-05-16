def roman_to_arabic(numeral):
    roman_dict = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, 'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500, 'CM': 900, 'M': 1000}
    arabic = 0
    i = 0
    while i < len(numeral):
        if i + 1 < len(numeral) and numeral[i:i+2] in roman_dict:
            arabic += roman_dict[numeral[i:i+2]]
            i += 2
        else:
            arabic += roman_dict[numeral[i]]
            i += 1
    return arabic

def arabic_to_roman(numeral):
    roman_dict = {1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 10: 'X', 40: 'XL', 50: 'L', 90: 'XC', 100: 'C', 400: 'CD', 500: 'D', 900: 'CM', 1000: 'M'}
    roman = ''
    for value, letter in sorted(roman_dict.items(), reverse=True):
        while numeral >= value:
            roman += letter
            numeral -= value
    return roman

while True:
    user_input = input("Enter a Roman numeral or an Arabic numeral (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    try:
        numeral = int(user_input)
        print(arabic_to_roman(numeral))
    except ValueError:
        numeral = user_input.upper()
        print(roman_to_arabic(numeral))
