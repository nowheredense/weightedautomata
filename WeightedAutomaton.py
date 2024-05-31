import itertools
import numpy
import heapq
import pickle

class WeightedAutomaton:
    # expects matrix_dict to be in usual format (recap here)
    # expects init_states and accept_state to be lists/tuples
    # TODO: make ring match ring in matrix_dict by default
    def __init__(self,matrix_dict,init_states,accept_states,alph=None,ring=QQ):
        """
        Create a WeightedAutomaton instance using transition matrices
        ``matrix_dict`` (as a dictionary in the form {letter: Sage matrix}),
        initial state vector ``init_states`` (as a list), and final state vector
        ``accept_states`` (as a list).

        The keys in ``matrix_dict`` should be strings.
        The values in ``matrix_dict`` must be square matrices, all of the same
        size, which must also match the lengths of ``init_states`` and
        ``accept_states``.

        You can optionally set the keyword-only argument ``ring`` to signal the
        ring the matrix coefficients are supposed to live in. Default is QQ.
        Currently, the constructor does nothing to check that your choice of
        ``ring`` actually matches the matrices you pass in.

        In the future, several alternative constructors will be provided to make
        this less clumsy to use.

        The constructor sets the properties self.transitions,
        self.initial_states, self.accepting_states, self.alphabet, self.ring,
        self.states, and self.size (the latter being the number of states).

        EXAMPLES::

            sage: mytransitions = {'0': matrix([[1/2,0,1], [0,1/2,0], [0,0,1]]),
                     '1': matrix([[0,1,0], [1/8,0,1/2], [0,0,1]]),
                     '2': matrix([[1,1,0], [0,0,0], [0,0,1]])}
            sage: initialstates = [1/2,1/2,0]
            sage: acceptingstates = [1,0,0]
            sage: myWA = WeightedAutomaton(mytransitions,initialstates,acceptingstates)
            sage: myWA
            Weighted finite automaton over the alphabet ['0', '1', '2'] and
            coefficients in Rational Field with 3 states

        """
        n = len(init_states)
        if n != len(accept_states):
            raise ValueError(f'{n} initial states != {len(accept_states)} accepting states')
        for m in matrix_dict.keys():
            # they should all be square matrices of same size
            if matrix_dict[m].dimensions() != (n,n):
                raise ValueError(f'matrix for {m} must be square and of dimension matching the number of states (yours is {matrix_dict[m].dimensions()})')
        self.transitions = matrix_dict
        # TODO: actually check matrix_dict is what it's supposed to be! And other sanity checks
        self.accepting_states = matrix(ring,accept_states).transpose()
        self.alphabet = list(matrix_dict.keys())
        self.initial_states = matrix(ring,init_states)
        self.ring = ring
        self.size = n
        self.states = list(range(0,n)) # TODO allow general state labels
        # the following is unused by default. It is only used by the symbolic()
        # constructor, which is planned in a future update, to
        # allow the user to access the symbolic variables used in a WA over SR.
        # (and it might be turned into a function anyway)
        ###self.vars = None

    def __repr__(self):
        return "Weighted finite automaton over the alphabet %s and coefficients in %s with %s states"%(self.alphabet,str(self.ring),self.size)

    # this one is intended for pickling the witness database, but it's also nice
    # to have a "plaintext" representation that doesn't need to pretty print
    def list(self):
        return [self.transitions,self.initial_states,self.accepting_states]

    def show(self):
        """
        Output a pretty-printed description of self.
        """
        for letter in self.alphabet:
            print(letter)
            show(table(self.transitions[letter]))
        print("Initial state distribution:", list(self.initial_states[0]))
        print("Final state distribution:", list(self.accepting_states.transpose()[0]))

    def __call__(self,string):
        """
        Return the acceptance probability of ``string``.
        """
        return self.prob(string)



    ################ UTILITY FUNCTIONS ####################
    
    @classmethod
    def _isint(self,obj):
        """
        Check if ``obj`` is an integer, either a Python int or a Sage Integer.
        """
        return (isinstance(obj,sage.rings.integer.Integer) or isinstance(obj,int))

    @classmethod
    def _pickletofile(self,object,name):
        """
        Store ``object`` in ``name``.pickle.
        """
        thefile = open(name+".pickle",'wb')
        pickle.dump(object,thefile)
        thefile.close()
        return

    @classmethod
    def _unpicklefile(self,filename):
        """
        Retrieve whatever is in ``filename``.pickle and return it.
        """
        thefile = open(filename+".pickle",'rb')
        theobj = pickle.load(thefile)
        thefile.close()
        return theobj

    @classmethod
    def _savewits(self,mywitnesses,filename):
        """
        Save the dictionary of WeightedAutomaton instances ``mywitnesses`` to
        ``filename.pickle``.
        ``mywitnesses`` is expected to be in the form retrieved by
        ``_loadwits()`` and will be saved in list format.
        """
        # we have to save in a dumb format because WeightedAutomaton instances are too fragile
        listSW = {}
        for k in SW.keys():
            listSW[k] = []
            for w in SW[k]:
                listSW[k].append(w.list())
        WeightedAutomaton._pickletofile(listSW,"witnesses")

    @classmethod
    def _loadwits(self,filename):
        """
        Load a witness dictionary from ``filename.pickle``, convert to a
        dictionary of ``WeightedAutomaton`` instances, and return the latter.
        """
        rawSW = WeightedAutomaton._unpicklefile("witnesses")
        newSW = {}
        for k in rawSW.keys():
            newSW[k] = []
            for w in rawSW[k]:
                newSW[k].append(WeightedAutomaton._listtoclass(w))
        return newSW

    @classmethod
    def _listtoclass(self,witness):
        """
        Convert a list or tuple ``witness'' specifying a WeightedAutomaton (as
        in output of ``list()``) to a WeightedAutomaton instance.
        """
        return WeightedAutomaton(witness[0], list(witness[1][0]),list(witness[2].transpose()[0]))

    @classmethod
    def _tuptostr(self,atuple):
        """
        Return a string that is everything in ``atuple`` concatenated.
        """
        return ''.join([str(t) for t in atuple])

    # return list of every string obtained by successively appending the characters listed in letters
    # letters is an iterable
    # optional: prefix everything with the string start
    @classmethod
    def _strorbit(self,letters,start=''):
        strings = [start]
        currstr = start
        for l in letters:
            currstr = currstr + l
            strings.append(currstr)
        return strings

    ################ END UTILITY FUNCTIONS ####################
   



    def transition_matrix(self,string):
        """
        Return transition matrix describing how the initial state distribution
        is updated after reading ``string``.
        """
        A = identity_matrix(self.size) # this allows to cover the case where string is empty
        for ch in string:
            if not str(ch) in self.alphabet:
                raise ValueError(f'{ch} not a valid letter of the alphabet')
            A = A*self.transitions[ch]
        return A

    def is_accepting(self,state):
        """
        Return True iff ``state`` (which can only be a number for now) is accepting,
        i.e., the final state distribution assigns a nonzero weight to it.
        If self.ring == SR, always return True because there's no meaningful way to say
        a state is *not* accepting in this case.
        """
        if self.ring == SR:
            # no way to meaningfully say if a state is accepting in this case
            return True 
        return (self.accepting_states[state][0] != 0)

    def read(self,string=''):
        """
        Return the probability distribution on states (given as a vector) after
        reading ``string``.
        If ``string`` is empty, this in effect just returns
        ``self.initial_states``.
        """
        return self.initial_states*self.transition_matrix(string)

    def trans_prob(self,state1,state2,string):
        """
        Return the probability of going from state ``state1`` to state ``state2``
        after reading the word ``string``.
        """
        if not (state1 in self.states and state2 in self.states):
            raise ValueError('you must specify two valid states')
        # i know this is ugly, but it's more general for when i eventually allow
        # arbitrary state labels
        index1 = self.states.index(state1)
        index2 = self.states.index(state2)
        # NOTE: although Sage starts row/column indexing from 1 as far as
        # permutations and such of matrices are concerned, it starts from 0 in
        # terms of their internal array representation. (!?)
        # Handle a single letter directly (maybe a bit faster since no new matrix created)
        if len(string) == 1:
            return self.transitions[string][index1][index2]
        else:
            return self.transition_matrix(string)[index1][index2]
    
    def swap_states(self,state1,state2):
        """
        Permute ``state1`` and ``state2``. Easy wrapper for ``permute_states()``
        so you don't have to create PermutationGroupElement instances every
        time. ``state1`` and ``state2`` should be the actual state labels
        (currently these are always numbers indexed from 0).
        """
        if not (state1 in self.states and state2 in self.states):
            raise ValueError('you must specify two valid states')
        # same ugliness comment as in trans_prob()
        # FIXME: whenever i fix permute_states(), it'll be unnecessary to add 1 here
        index1 = self.states.index(state1)+1
        index2 = self.states.index(state2)+1
        permgroup = PermutationGroup([(index1,index2)])
        self.permute_states(permgroup.gens()[0])

    # apply permutation, a PermutationGroupElement, to the set of states.
    # Actually (FIXME), it has to be a permutation of the set {1,...,self.size}, because
    # Sage insists on indexing matrix rows and columns starting at 1 and i can't
    # figure out how to take a permutation and just rename the underlying set.
    def permute_states(self,permutation):
        if type(permutation) != sage.groups.perm_gps.permgroup_element.PermutationGroupElement:
            raise TypeError('PermutationGroupElement expected')
        for a in self.alphabet:
            self.transitions[a] = \
                self.transitions[a].with_permuted_rows_and_columns(permutation,permutation)
        self.initial_states = self.initial_states.with_permuted_columns(permutation)
        self.accepting_states = self.accepting_states.with_permuted_rows(permutation)

    def set_transition(self,state1,state2,letter,value,reweight=False):
        """
        Change (probability of going from ``state1`` to ``state2`` when reading
        ``letter``) to ``value``.
        If ``reweight`` is True, rescale the weights of the other ``letter``-out-transitions
        from ``state1`` so they and ``value`` sum to 1. Only really makes sense
        to do if you're working with a PFA. If the other weights sum to 0,
        setting ``reweight`` to True will change ``value`` to 1 instead.
        """
        if not letter in self.alphabet:
            raise ValueError(f'letter {letter} not in alphabet')
        if not (state1 in self.states and state2 in self.states):
            raise ValueError('you must specify two valid states')
        # same ugliness comment as above. NOTE this time we don't need to add 1
        # to the state indices because we're directly accessing the internal
        # array structure of the matrices!
        index1 = self.states.index(state1)
        index2 = self.states.index(state2)
        self.transitions[letter][index1,index2] = value
        if reweight:
            restofrowsum = sum(self.transitions[letter][index1]) - value
            if restofrowsum != 0:
                for n in range(self.size): # for each entry in that row
                    if n != index2: # only change the entries we didn't set above
                        self.transitions[letter][index1,n] *= (1-value)/restofrowsum
            else:
                # this means the only way to make the row sum to 1 is actually to change
                # the [index1,index2] entry to 1 (overriding the caller's value)
                self.transitions[letter][index1,index2] = 1

    def set_initial(self,state,value,reweight=False):
        """
        Set the weight of ``state`` in the initial state distribution to
        ``value``.
        If ``reweight`` is set to True, do the same rescaling as in
        ``set_transition()`` but applied to the initial state vector.
        """
        theindex = self.states.index(state)
        self.initial_states[0,theindex] = value
        # FIXME: this code is basically duplicated from above. Figure out how to
        # farm out both into another method without running into immutability
        # issues.
        if reweight:
            restofrowsum = sum(self.initial_states[0]) - value
            if restofrowsum != 0:
                for n in range(self.size): # for each entry in the vector
                    if n != theindex: # only change the entries we didn't set above
                        self.initial_states[0,n] *= (1-value)/restofrowsum
            else:
                self.initial_states[0,theindex] = 1

    def set_final_vector(self,newvector_list):
        """
        QoL method to quickly change self.accepting_states to the column vector
        represented by ``newvector_list``.
        """
        if len(newvector_list) != self.size:
            raise TypeError('length of given vector != size of automaton')
        self.accepting_states = matrix(newvector_list).transpose()

    def normalize(self,letter=None,row=None,doinitialstates=True):
        """
        Normalize each row vector of self to sum to 1 (initial state vector and
        all rows of all transition matrices).
        Optional: specify which ``letter``'s transition matrix to do (default all);
                  specify which ``row`` to do (default all);
                  specify whether to also do the initial state vector (default yes)
        If a row sum is 0, leave it unchanged. Ditto the initial state vector.
        """
        if letter == None: doletters = self.alphabet
        else: doletters = [letter]
        if row == None: dorows = range(self.size)
        else: dorows = [row]
        if doinitialstates:
            thesum = sum(self.initial_states.coefficients())
            if thesum != 0:
                self.initial_states /= thesum
        for l in doletters:
            for r in dorows:
                thesum = sum(self.transitions[l][r].coefficients())
                if thesum != 0:
                    self.transitions[l][r] /= thesum

    def prob(self,string):
        """
        Return the acceptance probability of ``string``.
        """
        return (self.read(string)*self.accepting_states)[0][0]
        # (the [0][0] is because technically, the result of that product is a 1x1 matrix, not a number)

    def probs(self,wordlist):
        """
        Return ProbabilityList of acceptance probabilities of every string
        in ``wordlist``.
        """
        thelist = ProbabilityList()
        for w in wordlist:
            thelist[w] = self.prob(w)
        return thelist

    def probs_of_length(self,thelength):
        """
        Wrapper around ``probs()`` that returns a ProbabilityList of
        acceptance probabilities of every string of length ``thelength``.
        """
        wordlist = self.strings([thelength])
        return self.probs(wordlist)

    # return gap(string). This is the minimum difference of rho(string)
    # and rho(x) over all strings with |x|=|string| and x!=string.
    # Note that if self doesn't witness an upper bound for A_P(string), this
    # value will be nonpositive.
    # If specified, comparestrings is a list/tuple of strings against which to compare string
    # (If comparestrings is not specified, this function will generate the list
    # of strings of length |string| for comparison on every run, so if you're
    # going to be running this a lot of times you should really declare the
    # list of strings in advance and pass it to the function.)
    # If cutoff=True, just return 0 if the gap isn't positive. (Has no effect if we're over SR.)
    def gap(self,string,comparestrings=None,cutoff=False):
        if comparestrings == None: # compare with other strings of same length
            usestrings = self.strings([len(string)])
        else: # compare with strings given by caller
            usestrings = comparestrings
        # if over SR, we'll use a completely symbolic approach
        if self.ring == SR:
            funcs = []
            for s in usestrings:
                if s != string:
                    funcs.append(self.prob(theword) - self.prob(s))
            return min_symbolic(funcs)
        # otherwise we have a chance to be a bit more efficient
        else:
            thegap = 1 # initialize at max possible value because it's defined as a minimum
            myprob = self.prob(string)
            for s in usestrings:
                if s == string: continue # don't compare against self
                testgap = myprob - self.prob(s)
                # if myprob isn't maximal (and we're cutting off at gap 0), immediately end
                if testgap <= 0 and cutoff == True:
                    return 0
                if testgap < thegap:
                    thegap = testgap
            return thegap

    def orbit(self,letters,start=''):
        """
        Return ProbabilityList of acceptance probabilities of every string
        obtained by successively appending the characters listed in ``letters``.
        Optional: prefix everything with the string ``start``.
        """
        return self.probs(WeightedAutomaton._strorbit(letters,start))

    # TODO: this is obviously really wasteful when getting used in loops with a bajillion calculations!
    def strings(self,oflengths):
        """
        Return list of all possible strings drawn from self's alphabet, of
        lengths ``oflengths`` (an iterable).

        EXAMPLES::

            (Suppose A is a WeightedAutomaton over the alphabet ['0','1'].)

            sage: A.strings([2,3])
            ['00', '01', '10', '11', '000', '001', '010', '011', '100', '101', '110', '111']
        """
        thelist = []
        for i in oflengths:
            if not WeightedAutomaton._isint(i):
                raise TypeError(f'{i} is not a Sage or Python integer')
            thelist = thelist + [WeightedAutomaton._tuptostr(it) for it in itertools.product(self.alphabet,repeat=i)]
        return thelist

    # return True iff teststring has the highest prob among strings of its length
    # optional: among strings in the list/tuple comparestrings (if given)
    # this is faster in the best case than computing all probs and comparing, because it returns False as soon as it finds another string with prob >= prob of teststring
    def is_highest(self,teststring,comparestrings=None):
        if self.ring == SR:
            raise TypeError("symbolic probabilities can't be ordered")
        myprob = self.prob(teststring)
        if comparestrings == None: # compare with other strings of same length (wasteful for loops!)
            usestrings = self.strings([len(teststring)])
        else: # compare with strings given by caller
            usestrings = comparestrings
        for s in usestrings:
            if s == teststring: continue # don't compare against yourself
            if self.prob(s) >= myprob:
                return False
        return True
    
    # return True iff thestate is a nonaccepting state with no out-transitions to other states
    # no out-transitions <-> P_sigma(i,thestate) = 0 for each sigma and i!=thestate
    def is_dead_state(self,thestate):
        if self.is_accepting(thestate): return False
        return all([(self.trans_prob(thestate,i,sigma) == 0) for sigma in self.alphabet \
                for i in self.states if i != thestate])

    # return True iff self has a dead state
    def has_dead_state(self):
        return any([self.is_dead_state(s) for s in self.states])

    # return True iff self is a PFA as defined in, e.g., Salomaa (1969):
    # initial states and all matrix rows are stochastic vectors, accepting states is a 0-1 vector
    def is_pfa(self):
        # first check initial state distribution
        init = self.initial_states[0]
        if not (0 <= min(init) <= max(init) <= 1) or not (0 <= sum(init) <= 1):
            return False
        # now check matrices
        for mat in self.transitions.values():
            for r in range(self.size): # for each row in the transition matrix
                if not (0 <= min(mat[r]) <= max(mat[r]) <= 1) or not (0 <= sum(mat[r]) <= 1):
                    return False
        # finally, accepting states
        for entry in self.accepting_states.transpose()[0]:
            if entry != 0 and entry != 1:
                return False
        return True




#------------------------------------------------------------------

class ProbabilityList(dict):
    def to_sorted(self):
        """
        Return a copy of ``self``, sorted in descending order by probability.
        """
        return ProbabilityList(sorted(self.items(), key=lambda item: item[1], reverse=True))

    def slice(self,length,dosort=False):
        """
        Return a ``ProbabilityList`` (optionally sorted in descending order by
        probability, if ``dosort`` is set to True) consisting of the sub-dictionary of
        ``self``'s strings of length ``length``.
        """
        if not WeightedAutomaton._isint(length):
            raise TypeError('length must be a valid Sage or Python integer')
        if not length in self.lengths():
            raise ValueError(f'no strings of length {length} are present')
        newplist = ProbabilityList()
        for s in self.keys():
            if len(s) == length:
                newplist[s] = self[s]
        if dosort:
            newplist = newplist.to_sorted()
        return newplist

    def lengths(self):
        """
        Return set of string lengths represented in ``self``.
        """
        return set(len(s) for s in self.keys())

    # if oflength=None, return highest-prob strings among all strings in the dict's keys
    # otherwise, expects a Sage or Python integer. Return highest-prob strings among all strings of length oflength in the dict's keys
    # format in either case: a list whose first entry is the probability and whose subsequent entries are the strings of that probability
    # TODO: kind of awkwardly written. Harmonize with highest_gap().
    def highest_prob(self,oflength=None):
        # define usestrings to be the list of actual strings to compare
        if oflength == None:
            usestrings = list(self.keys())
        else:
            if not WeightedAutomaton._isint(oflength):
                raise TypeError('length must be a valid Sage or Python integer')
            if not oflength in self.lengths():
                raise ValueError(f'no strings of length {oflength} are present')
            usestrings = [k for k in self.keys() if len(k) == oflength]
        returnlist = []
        maxprob = max([self[w] for w in usestrings])
        returnlist.append(maxprob) # make the prob the first entry
        for w in usestrings:
            # list all strings with that probability
            if self[w] == maxprob:
                returnlist.append(w)
        return returnlist

    # Return gap between the two highest-prob strings of length oflength. If oflength
    # not specified, return gap between two highest-prob strings present in self.
    # Output: a list in the form [gap, highest-prob string, second-highest-prob string]
    # If numerical=True, give the gap as a numerical approximation (useful if
    # you want to get a quick idea of the value)
    def highest_gap(self,oflength=None,numerical=False):
        # if the caller specified a length, it must be a valid length
        if oflength != None:
            if not WeightedAutomaton._isint(oflength):
                raise TypeError('length must be a valid Sage or Python integer')
            if not oflength in self.lengths():
                raise ValueError(f'no strings of length {oflength} are present')
            # compare only between strings of length oflength
            compare = self.slice(oflength)
        else:
            # compare between strings of all lengths represented in self.
            # (obviously this will not actually produce a true 'gap' if more
            # than one length is present. Presumably the caller knows that.)
            compare = self
        # get 2 strings of highest prob, in order (using optimized library function)
        highest2 = heapq.nlargest(2, compare, key=compare.get)
        # prefix the list with the difference between their probs
        returncompare = compare[highest2[0]] - compare[highest2[1]]
        if numerical:
            return [returncompare.n()] + highest2
        else:
            return [returncompare] + highest2
    # TODO: it's weird to have highestgap() here but gap() in the PFA class. Maybe have both in both places, with the ones in PFA just being "compute the prob list, then run the PL func"?
    #       one issue would be the extra overhead of taking the slice when you already know everything is of the correct length





#SC = WeightedAutomaton._unpicklefile("stoch_complexity")
#NC = WeightedAutomaton._unpicklefile("nfa_complexity") # contains NFA complexities of binary strings of length <= 10
#SW = WeightedAutomaton._loadwits("witnesses")
