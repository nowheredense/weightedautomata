import itertools
import numpy
import urllib,json
import pickle


#------------------UTILITY FUNCTIONS------------------------

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


#------------------------------------------------------------------

SC = unpicklefile("stoch_complexity")
NC = unpicklefile("nfa_complexity")
SW = unpicklefile("witnesses")


#------------------------------------------------------------------

S2 = {} #set of binary strings of reasonable length
S2[0] = Set([])
for i in range(2,15):
    S2[i] = Set([tuptostr(it) for it in itertools.product([0,1],repeat=i)])
    S2[0] = S2[0].union(S2[i])

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

    def transition_matrix(self,string):
        A = identity_matrix(self.size)
        for ch in string:
            assert str(ch) in self.alphabet
            A = A*self.transitions[ch]
        return A

    # return acceptance probability of string
    def prob(self,string):
        return (self.initial_states*self.transition_matrix(string)*self.accepting_states)[0][0]
        # (the [0][0] is because technically, the result of that product is a 1x1 matrix, not a number)


    #returns gap(string). This is the minimum difference of rho(string)
    #and rho(x) over all strings with |x|=|string| and x!=string.
    #Note that if self doesn't witness an upper bound for A_P(string), this value will be negative.
    def gap(self,string):
        n = len(string)
        # pull only the probabilities of each other word of length n
        oflengthn = [tuptostr(it) for it in itertools.product(self.alphabet,repeat=n)]
        probsn = self.listprobs(oflengthn)
        #thelist = set([p[1] for p in self.list_probs(oflengthn) if p[0] != string])
        tocompare = set([probsn[k] for k in probsn.keys() if k != string])
        myprob = probsn[string]
        differencelist = [myprob - otherprob for otherprob in tocompare]
        return min(differencelist) #smallest such value is the gap, by definition

    # return ProbabilityList of acceptance probs wrt self of every string in wordlist
    # wordlist should be a list or tuple
    def listprobs(self,wordlist):
        thelist = ProbabilityList()
        for w in wordlist:
            thelist[w] = self.prob(w)
        return thelist

#------------------------------------------------------------------

# output of listprobs function
class ProbabilityList(dict):
    # return a /sorted/ ProbabilityList (in descending order by prob) consisting of the sub-dict of self's strings of length length
    def slice(self,length):
        return # obviously unimplemented


    # if oflength=None, return highest-prob strings among all strings in the dict's keys
    # otherwise, expects a Sage or Python integer. Return highest-prob strings among all strings of length oflength in the dict's keys
    # format in either case: a list whose first entry is the probability and whose subsequent entries are the strings of that probability
    def highest(self,oflength=None):
        # define usestrings to be the list of actual strings to compare
        if oflength == None:
            usestrings = list(self.keys())
        else:
            assert self._isint(oflength) # want this to be a single integer
            usestrings = [k for k in self.keys() if len(k) == oflength]
        returnlist = []
        maxprob = max([self[w] for w in usestrings])
        returnlist.append(maxprob) # make the prob the first entry
        for w in usestrings:
            if self[w] == maxprob: # list all strings with that probability
                returnlist.append(w)
        return returnlist
    
    #check if an object is an integer, either int or Sage integer
    def _isint(self,obj):
        return (isinstance(obj,sage.rings.integer.Integer) or isinstance(obj,int))
