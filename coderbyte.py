# My solutions for the coderbyte.com challanges


# Useful functions:
#  string.isalpha() - if all characters in the string are alphabets, abcd...
#  string.isnumeric - if all characters in the string are alphabets, 0123


def LongestWord(sen):

    """
    Have the function LongestWord(sen) take the sen parameter being passed and return the longest word in the string.
    If there are two or more words that are the same length, return the first word from the string with that length.
    Ignore punctuation and assume sen will not be empty. Words may also contain numbers, for example "Hello world123 567"
    """

    pl = '!"#$%&()*+, -./:;<=>?@[\]^_`{|}~'

    words = sen.split()
    words1 = [w.strip(pl) for w in words]  # clean words

    lengths = ['' for x in range(0, len(words1))]
    for i in range(0, len(words1)):
        lengths[i] = len(words1[i])

    max_index = lengths.index(max(lengths))

    return words1[max_index]
    # return max(words1, key=len)


print(LongestWord("fun&!! time007"))


def QuestionsMarks(strParam):

    """
    WARNING: This challange is questionable for the solution on Coderbyte.com
    Have the function QuestionsMarks(str) take the str string parameter, which will contain single digit numbers, letters,
    and question marks, and check if there are exactly 3 question marks between every pair of two numbers that add up to 10.
    If so, then your program should return the string true, otherwise it should return the string false. If there aren't any
    two numbers that add up to 10 in the string, then your program should return false as well. For example: if str is
    "arrb6???4xxbl5???eee5" then your program should return true because there are exactly 3 question marks between 6 and 4,
     and 3 question marks between 5 and 5 at the end of the string.
    """

    for i in range(0, len(strParam)):

        if strParam[i].isnumeric():
            # print('num')
            newstring = strParam[i + 1:]
            for j in range(0, len(newstring)):
                if newstring[j].isnumeric():
                    # check if two numbers sums to 10
                    if int(strParam[i]) + int(newstring[j]) == 10:
                        focusedstring = newstring[:j]
                        # print(focusedstring)
                        questions = [x == '?' for x in focusedstring]
                        if sum(questions) >= 3:
                            return True
                        else:
                            continue
                    else:
                        continue

                else:
                    continue

        else:
            # print('nonnum')
            continue

    return False


print(QuestionsMarks(input()))


def CodelandUsernameValidation(strParam):

    """
    Have the function CodelandUsernameValidation(str) take the str parameter being passed and determine if the string is a valid username according to the following rules:

    1. The username is between 4 and 25 characters.
    2. It must start with a letter.
    3. It can only contain letters, numbers, and the underscore character.
    4. It cannot end with an underscore character.

    If the username is valid then your program should return the string true, otherwise return the string false.
    """

    if len(strParam) < 4 or len(strParam) > 25:
        return False

    if strParam[0].isalpha() == False:
        return False

    if strParam[-1] == '_':
        return False

    for i in strParam:
        if i.isalpha() == True or i.isnumeric() == True or i == '_':
            continue
        else:
            return False

    return True

print(CodelandUsernameValidation(input()))


def BracketMatcher(strParam):

    """
    WARNING: doesn't work for double brackets e.g. "(hello (world))"
    Have the function BracketMatcher(str) take the str parameter being passed and return 1 if the brackets are correctly
    matched and each one is accounted for. Otherwise return 0. For example: if str is "(hello (world))", then the output should be 1,
    but if str is "((hello (world))" the the output should be 0 because the brackets do not correctly match up. Only "(" and ")" will
    be used as brackets. If str contains no brackets return 1.
    """

    if strParam.find('(') == -1 or strParam.find(')') == -1:
        return 1

    for i in range(len(strParam)):

        # check if opening is present
        if strParam[i] == '(':
            # check if closing is present
            newstring = strParam[i + 1:]
            for j in range(len(newstring)):
                if newstring[j] == ')' or newstring[j] == '(':
                    if newstring[j] == ')':
                        break
                    elif newstring[j] == '(':
                        return 0
                elif j == len(newstring) - 1:
                    return 0
                else:
                    continue

        elif strParam[i] == ')':
            newstring = strParam[:i]
            newstring = newstring[::-1]
            for j in range(len(newstring)):
                if newstring[j] == ')' or newstring[j] == '(':
                    if newstring[j] == '(':
                        break
                    elif newstring[j] == ')':
                        return 0
                elif j == len(newstring) - 1:
                    return 0
                else:
                    continue

        else:
            continue

    return 1


def BracketMatcher1(str):
    """
    Much more elegant code from example solutions on Coderbyte.com
    """

    count = 0
    for c in str:
        if c == '(':
            count += 1
        elif c == ')':
            count -= 1
        if count < 0: return 0

    return 1 if count == 0 else 0


print(BracketMatcher1("(hello (world))"))
print(BracketMatcher("(hello (world))"))
