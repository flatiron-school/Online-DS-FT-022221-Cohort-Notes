import numpy as np



def one_random_student(student_list, question = None):
    '''
    :param student_list: a list of students in any given class
    :return: a student to be called on
    '''

    student =  np.random.choice(student_list, 1)[0]
    print(student)
    print(question)

    
def pairs(student_list, question = None):
    '''
    :param student_list: a list of students in any given class
    :return: a student to be called on
    '''
    
    while len(student_list) >= 2:
        pair =  np.random.choice(student_list, 2, replace=False)
        print(pair)
        
        student_list.remove(pair[0])
        student_list.remove(pair[1])
        
    print(student_list)