import itertools
import numpy
import heapq
import urllib,json
import pickle


#------------------UTILITY FUNCTIONS------------------------

# (utility) check if an object is an integer, either int or Sage integer
def isint(obj):
    return (isinstance(obj,sage.rings.integer.Integer) or isinstance(obj,int))

# object: what to dump; name: string without ".pickle"
def pickletofile(object,name):
    thefile = open(name+".pickle",'wb')
    pickle.dump(object,thefile)
    thefile.close()
    return

# filename: string without ".pickle"
def unpicklefile(filename):
    thefile = open(filename+".pickle",'rb')
    theobj = pickle.load(thefile)
    thefile.close()
    return theobj

# NOTE: use dictionary NC for strings of length 2 through 10.
def nfacomplexity(string):
    theurl = "http://math.hawaii.edu/~bjoern/complexity-api/?string="+string
    response = urllib.request.urlopen(theurl)
    data = json.loads(response.read())
    return int(data.get('complexity'))

# returns a string that is everything in atuple concatenated
def tuptostr(atuple):
    return ''.join([str(t) for t in atuple])

# return list of every string obtained by successively appending the characters listed in letters
# letters is an iterable (incl string)
# optional: prefix everything with the string start
def strorbit(letters,start=''):
    assert len(letters) > 0
    strings = [start]
    currstr = start
    for l in letters:
        currstr = currstr + l
        strings.append(currstr)
    return strings

#------------------------------------------------------------------

SC = unpicklefile("stoch_complexity")
NC = unpicklefile("nfa_complexity")
SW = unpicklefile("witnesses")


#------------------------------------------------------------------

# this is too annoying to run every time, put it in the Jupyter notebook
#S2 = {} #set of binary strings of reasonable length
#S2[0] = Set([])
#for i in range(2,19):
#    S2[i] = Set([tuptostr(it) for it in itertools.product([0,1],repeat=i)])
#    S2[0] = S2[0].union(S2[i])

# expects lengths to be a list
def binstrings(lengths):
    returnset = Set([])
    for l in lengths:
        returnset = returnset.union(S2[l])
    return returnset

#------------------------------------------------------------------

# TODO: constructor from transitions (given as rules, not matrices)
#       fetch a single specified trans prob - and change one (w/o giving a whole new matrix)
#       change initial distribution, etc, without having to give a matrix object
class ProbabilisticFiniteAutomaton:
    # expects matrix_dict to be in usual format (recap here)
    # expects init_states and accept_state to be lists/tuples
    def __init__(self,matrix_dict,init_states,accept_states,ring=QQ):
        n = len(init_states)
        assert n == len(accept_states)
        for m in matrix_dict.keys():
            assert matrix_dict[m].dimensions() == (n,n) # they should all be square mats of same size
        # don't feel rn like allowing different rings: #self.transitions = {}
        self.transitions = matrix_dict
        #TODO: actually check matrix_dict is what it's supposed to be! And other sanity checks
        self.alphabet = list(matrix_dict.keys())
        self.initial_states = matrix(ring,init_states)
        self.accepting_states = matrix(ring,accept_states).transpose()
        self.size = n

    # TODO: (not worth it) get output formatted how i actually want (closest thing: converting to numpy)
    def show(self):
        for letter in self.alphabet:
            # print first row first, followed by others (doing this to align them)
            # casting to matrix again so it formats like a Sage matrix, not a tuple
            #print(letter,'\t',matrix(self.transitions[letter][0])) 
            #for r in range(self.size-1): # skipping first row (don't repeat)
            #    print('\t',matrix(self.transitions[letter][r+1]))
            print(letter)#,'\n',self.transitions[letter])
            show(table(self.transitions[letter]))
        print("Initial distribution:", list(self.initial_states[0]))
        print("Accepting states:", [i for i in range(self.size) if self.is_accepting(i)])

    def transition_matrix(self,string):
        A = identity_matrix(self.size)
        for ch in string:
            assert str(ch) in self.alphabet
            A = A*self.transitions[ch]
        return A

    # return True iff state (which can only be a number for now) is accepting
    def is_accepting(self,state):
        return (self.accepting_states[state][0] == 1)

    # return acceptance probability of string
    def prob(self,string):
        return (self.initial_states*self.transition_matrix(string)*self.accepting_states)[0][0]
        # (the [0][0] is because technically, the result of that product is a 1x1 matrix, not a number)

    # return ProbabilityList of acceptance probs wrt self of every string in wordlist
    # wordlist should be a list or tuple
    def probs(self,wordlist):
        thelist = ProbabilityList()
        for w in wordlist:
            thelist[w] = self.prob(w)
        return thelist

    #returns gap(string). This is the minimum difference of rho(string)
    #and rho(x) over all strings with |x|=|string| and x!=string.
    #Note that if self doesn't witness an upper bound for A_P(string), this value will be negative.
    def gap(self,string):
        n = len(string)
        # pull only the probabilities of each other word of length n
        oflengthn = self.strings([n])
        probsn = self.probs(oflengthn)
        return min([probsn[string] - probsn[otherstr] for otherstr in probsn.keys() if otherstr != string])

    # return ProbabilityList of acceptance probabilities of every string obtained by successively appending the characters listed in letters
    # letters is an iterable (incl string)
    # optional: prefix everything with the string start
    def orbit(self,letters,start=''):
        return self.probs(strorbit(letters,start))

    # return list of all possible strings drawn from self's alphabet, of lengths oflengths (list/tuple)
    def strings(self,oflengths):
        thelist = []
        for i in oflengths:
            assert isint(i)
            thelist = thelist + [tuptostr(it) for it in itertools.product(self.alphabet,repeat=i)]
        return thelist

    # return True iff teststring has the highest prob among strings of its length
    # this is faster than computing all probs and comparing because it returns False as soon as it finds another string with prob >= prob of teststring
    def ishighest(self,teststring):
        myprob = self.prob(teststring)
        if myprob == 0: return False # no need to even go on
        comparestrings = self.strings([len(teststring)]) #TODO: how much unnecessary overhead does this add?
        for s in comparestrings:
            if s == teststring: continue # don't compare against yourself
            if self.prob(s) >= myprob: return False
        return True

#------------------------------------------------------------------

class ProbabilityList(dict):
    # return a copy of self, sorted in descending order by probability
    def sorted(self):
        return ProbabilityList(sorted(self.items(), key=lambda item: item[1], reverse=True))

    # return a ProbabilityList (optionally sorted in descending order by prob, if dosort=True) consisting of the sub-dict of self's strings of length length
    def slice(self,length,dosort=True):
        assert isint(length) and length in self.lengths()
        newplist = ProbabilityList()
        for s in self.keys():
            if len(s) == length:
                newplist[s] = self[s]
        if dosort:
            newplist = newplist.sorted()
        return newplist

    # return list of string lengths represented in self
    def lengths(self):
        return list(set(len(s) for s in self.keys()))

    # if oflength=None, return highest-prob strings among all strings in the dict's keys
    # otherwise, expects a Sage or Python integer. Return highest-prob strings among all strings of length oflength in the dict's keys
    # format in either case: a list whose first entry is the probability and whose subsequent entries are the strings of that probability
    def highest(self,oflength=None):
        # define usestrings to be the list of actual strings to compare
        if oflength == None:
            usestrings = list(self.keys())
        else:
            assert isint(oflength) and oflength in self.lengths()
            usestrings = [k for k in self.keys() if len(k) == oflength]
        returnlist = []
        maxprob = max([self[w] for w in usestrings])
        returnlist.append(maxprob) # make the prob the first entry
        for w in usestrings:
            if self[w] == maxprob: # list all strings with that probability
                returnlist.append(w)
        return returnlist

    # return gap between the two highest-prob strings of oflength (which needs to be specified)
    # return format: [gap, highest-prob string, second-highest-prob string]
    def highestgap(self,oflength):
        assert isint(oflength) and oflength in self.lengths() # valid integer and valid length
        compare = self.slice(oflength,dosort=False)
        # get 2 strings of highest prob, in order (and want to avoid sorting, etc)
        highest2 = heapq.nlargest(2, compare, key=compare.get)
        # add the difference between their probs out front
        return [compare[highest2[0]] - compare[highest2[1]]] + highest2
