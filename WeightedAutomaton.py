import itertools
import numpy
import heapq
import pickle

class WeightedAutomaton(SageObject):
    def __init__(self, 
                 matrix_dict, 
                 init_states,
                 final_states,
#                 alph=None,
                 ring=QQ,
                 variables=None):
        """
        Return a WeightedAutomaton instance using transition matrices
        ``matrix_dict`` (as a dictionary in the form {letter: matrix}),
        initial state vector ``init_states`` (as a list, tuple, vector, or
        matrix with one row), and final state vector ``final_states`` (as a
        list, tuple, vector, or matrix with one column).

        The keys in ``matrix_dict`` will be coerced to strings and will become
        the letters of the alphabet the WeightedAutomaton reads from.
        The values in ``matrix_dict`` must be square matrices, all of the same
        size, which must also match the lengths of ``init_states`` and
        ``final_states``. They can be given as actual Sage matrices or as
        lists of lists, e.g., [[1,0,0],[0,1,0],[0,0,1]].

        You can optionally set the keyword-only argument ``ring`` to signal the
        ring the matrix coefficients are supposed to live in; this will result
        in all matrices and vectors having their rings changed to ``ring``.
        By default everything will be coerced to be over QQ (possibly subject to
        future change for greater flexibility).

        If ``variables`` is set then it should be a list or tuple of symbolic
        variables which presumably appear in the vectors or matrices. No check
        is done, so it is up to the caller to enforce this. If unset then
        ``self.vars`` will by default be set to ().

        The constructor sets the properties self.transitions, self.initial,
        self.final, self.alphabet, self.ring, self.states, self.vars, and
        self.size (the latter being the number of states).

        EXAMPLES::

            sage: mytransitions = {'0': [[1/2,0,1], [0,1/2,0], [0,0,1]],
                     '1': [[0,1,0], [1/8,0,1/2], [0,0,1]],
                     '2': [[1,1,0], [0,0,0], [0,0,1]]}
            sage: initialstates = [1/2,1/2,0]
            sage: acceptingstates = [1,0,0]
            sage: myWA = WeightedAutomaton(mytransitions,initialstates,acceptingstates,ring=SR)
            sage: myWA
            Weighted finite automaton over the alphabet ['0', '1', '2'] and
            coefficients in Symbolic Ring with 3 states

        """
        # this will work if init_states is a list, tuple, vector, or Sage matrix
        self.initial = deepcopy(matrix(ring,init_states))
        if self.initial.nrows() != 1:
            raise TypeError('invalid initial state vector (must be coerceable to a row vector)')
        # check if final_states is already a matrix (so we know whether to
        # transpose it or not)
        if isinstance(final_states,sage.matrix.matrix0.Matrix):
            self.final = deepcopy(matrix(ring,final_states))
        else:
            self.final = matrix(ring,final_states).transpose()
        n = self.initial.ncols()
        if self.final.dimensions() != (n,1):
            raise ValueError(f'final state vector should have dimensions ({n},1) (yours is {self.final.dimensions()})')
        newtransitions = {}
        for k in matrix_dict.keys():
            transfork = deepcopy(matrix(ring,matrix_dict[k]))
            # they should all be square matrices of same size
            if transfork.dimensions() != (n,n):
                raise TypeError(f'matrix for {k} must be square and of dimension matching the number of states (yours is {transfork.dimensions()} and should be ({n},{n}))')
            # this forces the keys to be strings, and we already forced the values to have the specified ring
            newtransitions[str(k)] = transfork
        self.transitions = newtransitions
        self.alphabet = list(self.transitions.keys())
        self.ring = ring
        self.size = n
        self.states = list(range(0,n)) # TODO allow general state labels
        if variables == None:
            self.vars = ()
        else:
            self.vars = tuple(variables)

    def __repr__(self):
        return "Weighted finite automaton over the alphabet %s and coefficients in %s with %s states"%(self.alphabet,str(self.ring),self.size)

    def __call__(self,string):
        """
        Return the acceptance probability of ``string``.
        """
        return self.prob(string)

    def __eq__(self,other):
        """
        Return True iff ``self`` and ``other`` have the same initial and final
        state vectors, alphabets, rings, states, and transition matrices.
        """
        return (self.initial == other.initial and
                self.final == other.final and
                self.transitions == other.transitions and
                self.states == other.states and
                self.ring == other.ring and
                self.alphabet == other.alphabet)
        # the ring /should/ be covered by the matrix equalities, but just to be on the safe side
        # similarly for alphabet

    def __pos__(self):
        return self

    def __neg__(self):
        """
        Return a WeightedAutomaton which is ``self`` with initial state vector
        negated. Hence (-self).prob(x) = -(self.prob(x)) for all strings
        x.
        """
        return WeightedAutomaton(self.transitions,-self.initial,
                                 self.final,ring=self.ring,
                                 variables=self.vars)

    def __add__(self,other):
        """
        Return the direct sum of ``self`` and ``other``, i.e.,
        self.direct_sum(other). See the latter function's documentation for more
        details.
        """
        return self.direct_sum(other)

    def __sub__(self,other):
        return self.__add__(other.__neg__())

    def __mul__(self,other):
        """
        If ``other`` is another WeightedAutomaton, return
        self.tensor_product(other); see the latter function for more details.
        If ``other`` is a number, return a WeightedAutomaton identical to
        ``self`` except with an initial state vector scaled by ``other``.
        """
        if isinstance(other,WeightedAutomaton):
            return self.tensor_product(other)
        else:
            # allow for scaling by stuff from different rings
            newring = self._biggerring(self.ring,base_ring(other))
            return WeightedAutomaton(self.transitions,
                                     newring(other)*self.initial.change_ring(newring),
                                     self.final, ring=newring,
                                     variables=self.vars)
    
    # this is to allow scalar multiplication in either order
    def __rmul__(self,other):
        return self.__mul__(other)


################################################################################

    def list(self):
        """
        Return a representation of ``self`` as a list, in the format [transition
        matrix dictionary, initial state vector, final state vector].

        A list in this format can be reconstituted into a WeightedAutomaton
        instance with ``from_list()``.
        """
        return [self.transitions,self.initial,self.final]
    
    def show(self):
        """
        Output a pretty-printed description of ``self``.
        """
        for letter in self.alphabet:
            print(letter)
            show(table(self.transitions[letter]))
        print("Initial state distribution:", list(self.initial[0]))
        print("Final state distribution:", list(self.final.transpose()[0]))

    def dump_as_strings(self):
        """
        Return a list representation of ``self`` in the format
            [matrix dict, initial vector, final vector]
        as with ``self.list()``, except that every matrix and vector is
        given by a list of coefficients, each of which is replaced by its string
        representation.

        The point of this is to be able to represent ``self`` in a very compact
        and portable way, for example to store on disk.
        A list as output by ``dump_as_strings()`` can be reconstituted into a
        WeightedAutomaton instance using ``WeightedAutomaton.from_dump()``.

        EXAMPLES::
        
            sage: A = WeightedAutomaton.constant(1/3,['a','b'])
            sage: A.dump_as_strings()
            [{'a': ['1/3', '2/3', '1/3', '2/3'], 'b': ['1/3', '2/3', '1/3', '2/3']},
            ['1/3', '2/3'],
            ['1', '0']]
        """
        matrixstrings = {}
        for l in self.alphabet:
            # we need to use dense_coefficient_list() to get every entry
            matrixstrings[l] = [str(c) for c in self.transitions[l].dense_coefficient_list()]
        return [matrixstrings, 
                [str(c) for c in self.initial.dense_coefficient_list()],
                [str(c) for c in self.final.dense_coefficient_list()]]


################################################################################
######################  CONSTRUCTORS  ##########################################
################################################################################

    # TODO: allow reconstructing symbolic variables
    @classmethod
    def from_list(self,thelist,ring=QQ):
        """
        Return a WeightedAutomaton instance reconstructed from a list
        ``thelist'' in the format returned by ``list()``.
        Optional: specify a ring ``ring`` into which to coerce all entries
        of all matrices and vectors (default QQ).
        """
        return WeightedAutomaton(thelist[0],thelist[1],thelist[2],ring=ring)

    @classmethod
    def from_dump(self,dumpedstring,ring=QQ):
        """
        Return a WeightedAutomaton instance reconstructed from ``dumpedstring``,
        a string describing a WA exactly in the format output by
        ``dump_as_strings()``.
        Optional: specify a ring ``ring`` into which to coerce all entries
        of all matrices and vectors (default QQ).
        """
        preparetrans = {}
        mysize = len(dumpedstring[1])
        for k in dumpedstring[0].keys():
            preparetrans[k] = matrix(mysize,mysize,
                    [ring(i) for i in dumpedstring[0][k]])
        return WeightedAutomaton(preparetrans,
            [ring(c) for c in dumpedstring[1]],
            [ring(c) for c in dumpedstring[2]], ring=ring)

    @classmethod
    def madic(self,homo,reverse=False):
        """
        Return an m-adic PFA, a la Salomaa/Turakainen.
        Input: ``homo``, a dict giving a monoid homomorphism on the set of
        finite strings over some alphabet. Keys of ``homo`` are
        generators of the language (letters of the alphabet) and values are
        their images. Every letter must (at present) be castable to an int.

        The WeightedAutomaton returned will read from the alphabet ['0', ...,
        'm'] where 'm-1' is the largest key specified in ``homo``.
        If ``reverse`` is False, it will have 3 states, and its acceptance
        probability function maps every string s to the number 0.phi(s) in base
        m, where phi is the homomorphism represented by ``homo``.
        If ``reverse`` is True, it will have 2 states, and its acceptance
        probability function maps every string s to the number 0.phi(rev(s)) in
        base m, where rev(s) is the reversal of s.

        If the image of some letter k <= m is not given in ``homo``, the
        homomorphism will be assumed to send 'k' to itself, i.e., the dictionary
        entry 'k': 'k' will be assumed. However, at present, if a letter 'a' is
        ever used in any of ``homo.values()``, then 'b' must appear as a key in
        ``homo`` for some b >= a.
        """
        # return the number "0.string" in base "base"
        def madic_expansion(string, base):
            value = QQ(0)
            base = QQ(base)
            counter = 0 # we need to keep track of the position
            while counter < len(string):
                if not int(string[counter]) < base:
                    raise ValueError(f'{string} is not a valid {base}-adic string')
                value += QQ(string[counter]) / (base**(QQ(counter)+1))
                counter += 1
            return value
        #-----------------------------------------
        m = QQ(max([int(a) for a in homo.keys()])+1)
        alph = [str(n) for n in range(m)]
        themats = {} # matrix dict to generate
        for letter in alph:
            if letter in homo.keys():
                phi = madic_expansion(homo[letter],m)
                malpha = m**(-QQ(len(homo[letter])))
            else: # if the user didn't specify an image of letter, assume letter |-> letter
                phi = madic_expansion(letter,m)
                malpha = m**(-1)
            if reverse == False:
                themats[letter] = [[malpha, 1-malpha-phi, phi],
                                   [0,1,0],
                                   [0,0,1]]
            else:
                themats[letter] = [[1-phi,phi],
                                   [1-phi-malpha,phi+malpha]]
        if reverse == False:
            init = [1,0,0]; final = [0,0,1]
        else:
            init = [1,0]; final = [0,1]
        return WeightedAutomaton(themats, init, final, ring=QQ)

    @classmethod
    def constant(self,value,alph):
        """
        Return a 2-state WeightedAutomaton over alphabet ``alph`` whose
        acceptance probability function is identically equal to ``value``. If
        ``value`` is in the interval [0,1], the result will be a PFA.
        See Paz (1971), p. 146.
        """
        myinit = [value,1-value]
        myfinal = [1,0]
        # every transition matrix is the same, with all rows equal to myinit
        mydict = dict(zip(alph,
                      [matrix(base_ring(value),[myinit]*2)] * len(alph)))
        return WeightedAutomaton(mydict,myinit,myfinal,
                                 ring=base_ring(value))

    @classmethod
    def identity(self,nstates=1,alph=['0'],ring=QQ):
        """
        Return a WeightedAutomaton with ``nstates`` (default 1) states over
        alphabet ``alph`` (default ['0']) and ring ``ring`` (default QQ), all of
        whose transition matrices are the identity matrix.
        Its initial and final vectors will both be (1,0,...,0).
        The output is always a PFA.
        """
        if nstates <= 0:
            raise ValueError('a WeightedAutomaton must have at least one state')
        initandfinal = [1]+[0]*(nstates-1)
        return WeightedAutomaton(dict(zip(alph,
                                 [identity_matrix(nstates)]*len(alph))),
                                 initandfinal, initandfinal, ring=ring)

    @classmethod
    def symbolic(self,nstates,alph=['0','1'],varname='p',pfa=False,finalvector=None):
        """
        Return a WeightedAutomaton with ``nstates`` states over the alphabet
        ``alph`` such that every entry is a different symbolic variable
        generated by the name ``varname``.
        The output WeightedAutomaton will have its ``.vars`` property set so
        that the caller can access the list of variables.

        If ``pfa`` is set to True, the initial state vector and each transition
        matrix will be made generalized stochastic, in that the last entry of
        each row vector ends with 1 minus the sum of the rest of the entries. 
        In this case the caller can optionally set ``finalvector`` to what they
        want the final/accepting state vector to be (default: only the first
        state is accepting). The ``finalvector`` argument is ignored when
        ``pfa`` is False.

        EXAMPLES::

            sage: A = WeightedAutomaton.symbolic(3)
            sage: A.list()
            [
            {'0': [p0 p1 p2]                                    
            [p3 p4 p5]                                          
            [p6 p7 p8], '1': [ p9 p10 p11]                 [p21]
            [p12 p13 p14]                                  [p22]
            [p15 p16 p17]}                , [p18 p19 p20], [p23]
            ]

            You can access the variable generator through the "vars" property, for
            example to perform substitutions:

            sage: p=A.vars; S=A.subs(dict(zip(p,[1]*len(p))))
            sage: S.list()
            [
            {'0': [1 1 1]                      
            [1 1 1]                            
            [1 1 1], '1': [1 1 1]           [1]
            [1 1 1]                         [1]
            [1 1 1]}             , [1 1 1], [1]
            ]

            Setting "pfa=True" forces the automaton to be generalized stochastic:

            sage: B=WeightedAutomaton.symbolic(2,['1','2'],pfa=True,varname='b',finalvector=[0,1])
            sage: B.list()
            [
            {'1': [     b0 -b0 + 1]                                          
            [     b1 -b1 + 1], '2': [     b2 -b2 + 1]                     [0]
            [     b3 -b3 + 1]}                       , [     b4 -b4 + 1], [1]
            ]

            Even when setting "pfa=True", "is_pfa()" returns False because it
            can't determine if the variable entries are positive.
        """
        # Start by computing the number of variables to use and setting up the
        # variable generator
        if pfa == False:
            # nstates entries in each of two vectors of length nstates and
            # in each row of each of |alph| matrices of size nstates*nstates
            numvars = 2*nstates + len(alph)*nstates*nstates
        else:
            # nstates-1 entries in each of one vector of length nstates and
            # in each of the nstates rows of each of |alph| matrices
            numvars = (nstates-1) + len(alph)*nstates*(nstates-1)
        R = PolynomialRing(QQ,numvars,varname)
        var = R.gens()
        # now to actually build the automaton
        mats = {}
        if pfa == False:
            for i in range(len(alph)):
                # each matrix takes nstates*nstates variables
                mats[alph[i]] = matrix(SR,nstates,
                                 var[i*nstates*nstates:(i+1)*nstates*nstates])
            # At this point, we've used up var[0:len(alph)*nstates^2].
            usedvars = len(alph)*nstates*nstates
            # Use the last two sets of nstates variables for the initial and
            # final vectors:
            vi = var[usedvars:usedvars+nstates]
            vf = var[usedvars+nstates:]
        else:
            # each matrix takes nstates*(nstates-1) variables
            varincrement = nstates*(nstates-1)
            for i in range(len(alph)):
                mats[alph[i]] = matrix(SR,nstates,nstates-1,
                                 var[i*varincrement:(i+1)*varincrement])
                # complete the matrix by making it (generalized) stochastic
                rightcolumn = column_matrix(
                              [1-sum(mats[alph[i]].rows()[j].coefficients()) 
                               for j in range(nstates)])
                mats[alph[i]] = mats[alph[i]].augment(rightcolumn)
            # At this point, we've used up var[0:len(alph)*nstates*(nstates-1)].
            # Use the last set of nstates-1 variables for the initial vector,
            # and whatever the user specified for the final vector:
            vi = list(var[len(alph)*nstates*(nstates-1):])
            vi += [1-sum(vi)]
            if finalvector == None:
                vf = [1] + [0]*(nstates-1)
            else:
                # if the caller passes in garbage, the constructor will catch it
                vf = finalvector
        return WeightedAutomaton(mats,vi,vf,ring=SR,variables=var)


################################################################################
#################### UTILITY FUNCTIONS #########################################
################################################################################
    
    @classmethod
    def _isint(self,obj):
        """
        Check if ``obj`` is either a Python int or a Sage Integer.
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
        ``filename``. Will be a zlib-encoded dict, readable by the Sage function
        loads(), where each WA is given by its representation via
        ``dump_as_strings()``. 
        ``mywitnesses`` is expected to be in the form retrieved by
        ``_loadwits()`` and will be saved in list format.
        """
        # we have to save in a dumb format because WeightedAutomaton instances are too fragile
        listSW = {}
        for k in mywitnesses.keys():
            listSW[k] = []
            for w in mywitnesses[k]:
                listSW[k].append(w.dump_as_strings())
        with open(filename,"wb") as f:
            f.write(dumps(listSW))

    @classmethod
    def _loadwits(self,filename):
        """
        Load a witness dictionary from ``filename`` (expected in the format
        output by ``_savewits()``), convert to a
        dictionary of WeightedAutomaton instances, and return the latter.
        """
        with open(filename,"rb") as f:
            rawSW = loads(f.read())
        newSW = {}
        for k in rawSW.keys():
            newSW[k] = []
            for w in rawSW[k]:
                newSW[k].append(WeightedAutomaton.from_dump(w,ring=QQ))
        return newSW

    # TODO: docstring
    @classmethod
    def _biggerring(self,ring1,ring2):
        if ring1.is_subring(ring2):
            return ring2
        elif ring2.is_subring(ring1):
            return ring1
        else:
            return SR


################################################################################
###################### CALCULATING PROBABILITIES ###############################
################################################################################

    def transition_matrix(self,string):
        """
        Return transition matrix describing how the initial state distribution
        is updated after reading ``string``.
        """
        A = identity_matrix(self.size) # this allows to cover the case where ``string`` is empty
        for ch in string:
            if not str(ch) in self.alphabet:
                raise ValueError(f'{ch} is not a valid letter of the alphabet {self.alphabet}')
            A *= self.transitions[ch]
        return A

    def read(self,string=''):
        """
        Return the probability distribution on states (given as a vector) after
        reading ``string``.
        If ``string`` is empty, this in effect just returns
        ``self.initial``.
        """
        return self.initial*self.transition_matrix(string)

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
   
    def prob(self,string):
        """
        Return the acceptance probability of ``string`` with respect to
        ``self``. This is the product of ``self``'s initial state vector,
        the transition matrices for the letters of ``string`` (in order), and
        ``self``'s final state vector.
        """
        return (self.read(string)*self.final)[0][0]
        # (the [0][0] is because technically, the result of that product is a 1x1 matrix, not a number)

    def probs(self,wordlist):
        """
        Return ProbabilityList of acceptance probabilities with respect to
        ``self`` of every string in ``wordlist``.
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

    def gap(self,string,comparestrings=None,cutoff=False):
        """
        Return the gap of ``string`` with respect to ``self``. This is the
        least difference between self.prob(``string``) and self.prob(x) where
        x ranges over all other strings of the same length as ``string``.
        (See Gill 2024 for a formal definition.)
        This value is positive iff ``string`` has the highest acceptance
        probability among strings of its length.
        
        If specified, ``comparestrings`` is a list of strings against which to
        compare ``string``. If ``comparestrings`` is not specified, ``gap()``
        will generate self.strings([len(``string``)]) on every run.
        Generating this set in advance and passing it to ``gap()`` is useful to
        save time in loops.
        If ``cutoff`` is set to True, just return 0 if the gap isn't positive.
        (Has no effect if self.ring == SR.)
        """
        if comparestrings == None: # compare with other strings of same length
            usestrings = self.strings([len(string)])
        else: # compare with strings given by caller
            usestrings = comparestrings
        myprob = self.prob(string)
        # if over SR, in general we have to use a completely symbolic approach
        if self.ring == SR:
            funcs = []
            for s in usestrings:
                if s != string:
                    funcs.append(myprob - self.prob(s))
            return min_symbolic(funcs)
        # otherwise we have a chance to be a bit more efficient
        else:
            thegap = self.ring(1) # initialize at max possible value because it's defined as a minimum
            for s in usestrings:
                if s == string: continue # don't compare against yourself
                testgap = myprob - self.prob(s)
                # if myprob isn't maximal (and we're cutting off at gap 0), immediately end
                if testgap <= 0 and cutoff == True:
                    return self.ring(0)
                if testgap < thegap:
                    thegap = testgap
            return thegap

    def orbit(self,letters,start=''):
        """
        Return ProbabilityList of acceptance probabilities of every string
        obtained by successively appending the characters listed in ``letters``
        to ``start``.
        """
        stringsinorbit = [start]
        currstr = start
        for l in letters:
            currstr += l
            stringsinorbit.append(currstr)
        return self.probs(stringsinorbit)

    def strings(self,oflengths):
        """
        Return list of all possible strings drawn from self's alphabet, of
        lengths given in the list ``oflengths``.

        EXAMPLES::

            (Suppose A is a WeightedAutomaton over the alphabet ['0','1'].)

            sage: A.strings([2,3])
            ['00', '01', '10', '11', '000', '001', '010', '011', '100', '101', '110', '111']

            If B is a WeightedAutomaton over ['0', '12'], strings() will
            represent each string as a tuple in order to avoid the ambiguity of
            strings like '012':
            
            sage: B.strings([2])
            [('0', '0'), ('0', '12'), ('12', '0'), ('12', '12')]
        """
        thelist = []
        for i in oflengths:
            if not WeightedAutomaton._isint(i):
                raise TypeError(f'{i} is not a Sage or Python integer')
            # if the max length of a letter is 1, make the outputs into actual
            # strings, because they're unambiguous and easier to work with.
            # Otherwise, each string needs to be represented as a list.
            if self.letter_length() == 1:
                morestrings = [''.join([str(c) for c in it]) 
                               for it in itertools.product(self.alphabet,repeat=i)]
            else:
                morestrings = list(itertools.product(self.alphabet,repeat=i))
            thelist = thelist + morestrings
        return thelist

    def is_highest(self,teststring,comparestrings=None):
        """
        Return True iff ``teststring`` has the unique highest probability of
        acceptance by ``self`` among all strings of length len(``teststring``).
        This function is faster on average than computing all probabilities and
        then comparing, because it immediately returns False when it finds
        another string with higher probability than ``teststring``.

        Optionally, you can specify a list ``comparestrings`` of strings against
        which to compare ``teststring``'s probability. (This can save time in
        loops.)
        """
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

################################################################################
############################# PROPERTIES #######################################
################################################################################

    def is_accepting(self,state):
        """
        Return True iff ``state`` (which can only be a number for now) is accepting,
        i.e., the final state distribution assigns a nonzero weight to it.
        """
        return (self.final[state][0] != 0)

    def is_dead_state(self,thestate):
        """
        Return True iff ``thestate`` is a nonaccepting state with no
        out-transitions to other states (in other words, the weight of
        ``thestate`` in ``self.final`` is 0, and all transitions from
        ``thestates`` to other states have weight 0).
        """
        if not thestate in self.states:
            raise ValueError('invalid state')
        if self.is_accepting(thestate): return False
        return all([(self.trans_prob(thestate,i,sigma) == 0) for sigma in self.alphabet \
                for i in self.states if i != thestate])

    def has_dead_state(self):
        """
        Return True iff ``self`` has a dead state, a nonaccepting state with no
        nonzero out-transitions.
        """
        return any([self.is_dead_state(s) for s in self.states])

    def is_pfa(self):
        """
        Return True iff ``self`` is a PFA (probabilistic finite-state automaton)
        as defined in, e.g., the book by Salomaa (1969).
        That is, return True iff the initial state vector and every row of every
        transition matrix of ``self`` are stochastic vectors, and the final
        state vector's entries are all either 0 or 1.
        """
        # first check initial state distribution
        init = self.initial.dense_coefficient_list()
        if not (0 <= min(init) <= max(init) <= 1) or not (sum(init) == 1):
            return False
        # now check matrices
        for mat in self.transitions.values():
            for r in range(self.size): # for each row in the transition matrix
                if not (0 <= min(mat[r]) <= max(mat[r]) <= 1) or not (sum(mat[r]) == 1):
                    return False
        # finally, final states
        for entry in self.final.dense_coefficient_list():
            if entry != 0 and entry != 1:
                return False
        return True

    def is_bistochastic(self):
        """
        Return True iff every transition matrix of ``self`` is bistochastic,
        i.e., both row and column stochastic.
        """
        return all([m.is_bistochastic() for m in self.transitions.values()])

    def is_actual(self):
        """
        Return True iff ``self`` is an actual automaton as defined by Rabin,
        i.e., is a PFA and all transition probabilities are strictly positive.
        """
        return (self.is_pfa() and all([all([p>0 for p in m.dense_coefficient_list()]) \
                    for m in self.transitions.values()]))

    def letter_length(self):
        """
        Return the longest length of a single letter in ``self``'s alphabet.
        """
        return max([len(l) for l in self.alphabet])

    def initial_weight(self,thestate):
        """
        Return the weight of ``thestate`` in the initial state vector of
        ``self``.
        """
        if not thestate in self.states:
            raise ValueError(f'invalid state {thestate}')
        return self.initial[0][thestate]

    def final_weight(self,thestate):
        """
        Return the weight of ``thestate`` in the final state vector of
        ``self``.
        """
        if not thestate in self.states:
            raise ValueError(f'invalid state {thestate}')
        return self.final[thestate,0]

################################################################################
########################## MODIFICATIONS #######################################
################################################################################

    def change_ring(self,newring):
        """
        Return new WeightedAutomaton obtained by coercing each of ``self``'s
        transition matrices and initial/final vectors to be over ring
        ``newring``.
        """
        return WeightedAutomaton(self.transitions,self.initial,
                                 self.final,ring=newring,variables=self.vars)

    def add_letter(self,letter,newtransitions=None):
        """
        Return new WeightedAutomaton which is ``self`` with alphabet augmented
        by ``letter`` and with the transition matrix for ``letter`` given by
        ``newtransitions``. If ``newtransitions`` is unspecified, it will be set
        to the identity matrix of size ``self``.size.
        If ``letter`` is already in ``self``'s alphabet, throw an error.
        """
        if str(letter) in self.alphabet:
            raise IndexError(f'letter {letter} is already in the alphabet')
        newdict = deepcopy(self.transitions)
        if newtransitions == None:
            newdict[letter] = identity_matrix(self.size)
        else:
            # we'll let the constructor handle if ``newtransitions`` isn't what
            # it's supposed to be
            newdict[letter] = newtransitions
        return WeightedAutomaton(newdict,self.initial,self.final,
                                 ring=self.ring,variables=self.vars)

    def delete_letter(self,letter):
        """
        Return new WeightedAutomaton which is ``self`` with ``letter`` removed
        from the alphabet along with its transition matrix.
        """
        if not letter in self.alphabet:
            raise IndexError(f'{letter} is not in the alphabet')
        newdict = deepcopy(self.transitions)
        newdict.pop(letter)
        return WeightedAutomaton(newdict,self.initial,self.final,
                                 ring=self.ring,variables=self.vars)

    def swap_states(self,state1,state2):
        """
        Return new WeightedAutomaton consisting of ``self`` with ``state1`` and
        ``state2`` switched. These inputs can be either the actual indices of
        states or the state labels (not yet implemented). This operation has no
        effect on acceptance probabilities.
        """
        if not (state1 in self.states and state2 in self.states):
            raise ValueError('you must specify two valid states')
        # same ugliness comment as in trans_prob(). And redundant for now.
        index1 = self.states.index(state1)
        index2 = self.states.index(state2)
        newdict = deepcopy(self.transitions)
        for m in newdict.values():
            m.swap_rows(index1,index2)
            m.swap_columns(index1,index2)
        return WeightedAutomaton(newdict,
                                 self.initial.with_swapped_columns(index1,index2),
                                 self.final.with_swapped_rows(index1,index2),
                                 ring=self.ring,variables=self.vars)

    # TODO: flesh out as indicated
    def add_states(self,nstates=1):
        """
        Return new WeightedAutomaton which is ``self`` with ``nstates`` new
        states added after all existing states.
        In the future, this will let you insert the new states at any position,
        and will support arbitrary state labels.
        The new states will have initial and final weight 0, no other state will
        transition to them, they will not transition to any other state, and
        each will transition to itself with probability 1 when reading any
        letter. Hence the output WeightedAutomaton has the same acceptance
        probability function as ``self``.
        """
        if nstates < 0:
            raise ValueError('cannot add a negative number of states')
        newinitial = self.initial.augment(matrix(self.ring,[0]*nstates))
        newfinal = self.final.stack(column_matrix([0]*nstates))
        newdict = {}
        for a in self.alphabet:
            newdict[a] = self.transitions[a].block_sum(
                            identity_matrix(nstates,nstates))
        return WeightedAutomaton(newdict,newinitial,newfinal,
                                 ring=self.ring,variables=self.vars)

    def delete_states(self,statelist):
        """
        Return new WeightedAutomaton which is ``self`` with each of the states
        in ``statelist`` deleted.
        """
        if not all([s in self.states for s in statelist]):
            raise IndexError(f'{statelist} contains invalid states')
        indexlist = [self.states.index(s) for s in statelist] # for when i allow arbitrary state labels
        newinit = self.initial.delete_columns(indexlist)
        newfinal = self.final.delete_rows(indexlist)
        newtrans = {}
        for a in self.alphabet:
            newtrans[a] = (self.transitions[a].delete_rows(indexlist)
                .delete_columns(indexlist))
        return WeightedAutomaton(newtrans,newinit,newfinal,ring=self.ring,
                                 variables=self.vars)

    # TODO: decide if it's too fussy to allow to reweight in every one of these
    # methods. Maybe normalize[d]() should be the only place to do it.
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
        castvalue = self.ring(value)
        self.transitions[letter][index1,index2] = castvalue
        if reweight:
            restofrowsum = sum(self.transitions[letter][index1]) - castvalue
            if restofrowsum != 0:
                for n in range(self.size): # for each entry in that row
                    if n != index2: # only change the entries we didn't set above
                        self.transitions[letter][index1,n] *= (1-castvalue)/restofrowsum
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
        if not state in self.states:
            raise IndexError(f'{state} is not a valid state')
        theindex = self.states.index(state)
        castvalue = self.ring(value)
        self.initial[0,theindex] = castvalue
        # FIXME: this code is basically duplicated from above. Figure out how to
        # farm out both into another method without running into immutability
        # issues.
        if reweight:
            restofrowsum = sum(self.initial[0]) - castvalue
            if restofrowsum != 0:
                for n in range(self.size): # for each entry in the vector
                    if n != theindex: # only change the entries we didn't set above
                        self.initial[0,n] *= (1-castvalue)/restofrowsum
            else:
                self.initial[0,theindex] = 1

    def set_final(self,state,value):
        """
        Set the weight of ``state`` in the final state vector to ``value``.
        """
        if not state in self.states:
            raise IndexError(f'{state} is not a valid state')
        theindex = self.states.index(state)
        self.final[theindex,0] = self.ring(value)

    def with_initial_vector(self,newinitial):
        """
        Return a copy of ``self`` with initial state vector replaced with
        ``newinitial``.
        """
        return WeightedAutomaton(self.transitions,newinitial,self.final,
                                 ring=self.ring,variables=self.vars)

    def with_final_vector(self,newfinal):
        """
        Return a copy of ``self`` with final state vector replaced with
        ``newfinal``.
        """
        return WeightedAutomaton(self.transitions,self.initial,newfinal,
                                 ring=self.ring,variables=self.vars)

    # TODO: allow more flexibility. Maybe have a separate normalize_all()
    # and then separate methods where you need to specify exactly which things
    # you want to normalize.
    # Also, what about self.final?
    def normalized(self,letters=None,rows=None,doinitialstates=True):
        """
        Normalize each row vector of self to sum to 1 (initial state vector and
        all rows of all transition matrices).
        Optional: specify which ``letters``'s transition matrices to do (default all);
                  specify which ``rows`` to do (default all);
                  specify whether to also do the initial state vector (default yes)
        ``letters`` and ``rows`` should be lists.
        If a row sum is 0, leave it unchanged. Ditto the initial state vector.

        If ``self`` is a PFA, the output is identical to ``self``.
        """
        if letters == None:
            letters = self.alphabet
        if rows == None:
            rows = range(self.size)
        newinitial = deepcopy(self.initial)
        newdict = deepcopy(self.transitions)
        if doinitialstates:
            thesum = sum(self.initial.coefficients())
            if thesum != 0:
                newinitial /= thesum
        for l in letters:
            for r in rows:
                thesum = sum(self.transitions[l][r].coefficients())
                if thesum != 0:
                    newdict[l][r] /= thesum
        return WeightedAutomaton(newdict,newinitial,self.final,ring=self.ring,
                                 variables=self.vars)

    def subs(self, *args, **kwds):
        """
        Substitute values for the variables used in the initial and final
        vectors and transition matrices of ``self``, and return the resulting
        WeightedAutomaton.
       
        The output will have the same ring and variables as ``self``.

        As with Sage's variable substition for matrices, the arguments are
        passed unchanged to the method ``subs`` of each matrix and vector.

        EXAMPLES::

            sage: a,b=var('a,b')
            sage: A=WeightedAutomaton({'0': [[1,1-a]]*2, '1': [[1,1-b]]*2}, 
                                      [a,0], [0,b], ring=SR)
            sage: B=A.subs(b=2)
            sage: B.list()
            [
            {'0': [     1 -a + 1]                    
            [     1 -a + 1], '1': [ 1 -1]         [0]
            [ 1 -1]}                     , [a 0], [2]
            ]
        """
        newdict = dict(zip(self.alphabet, [m.subs(*args,**kwds) for m in
                                           self.transitions.values()]))
        return WeightedAutomaton(newdict, self.initial.subs(*args,**kwds),
                                 self.final.subs(*args,**kwds), ring=self.ring,
                                 variables=self.vars)

################################################################################
########################## ALGEBRAIC OPERATIONS ################################
################################################################################

    def transpose(self):
        """
        Return the transpose of ``self``, that is, the WeightedAutomaton which
        is ``self`` with initial and final state vectors switched and all
        transition matrices transposed.
        
        The acceptance probability of x with respect to ``self.transpose()`` is
        equal to the acceptance probability of the reversal of x with respect to
        ``self`` for every string x.

        Note that the output is *not* necessarily a PFA when ``self`` is.
        """
        newdict = {}
        for a in self.alphabet:
            newdict[a] = self.transitions[a].transpose()
        return WeightedAutomaton(newdict, self.final.transpose(),
                                 self.initial.transpose(), ring=self.ring,
                                 variables=self.vars)

    def complement(self):
        """
        Return a copy of ``self`` whose final state vector is replaced by a
        vector of 1s subtracted from ``self``.final. The result has the same
        size as ``self``.
       
        If ``self`` is a PFA, the output M is also a PFA and satisfies 
            M.prob(x) = 1 - ``self``.prob(x)
        for every string x.
        """
        return WeightedAutomaton(self.transitions, self.initial,
                                 matrix.ones(self.ring,self.size,1) - self.final, 
                                 ring=self.ring,variables=self.vars)

    def scaled(self,scalefactor):
        """
        Return the WeightedAutomaton which is the tensor product of ``self``
        with a 2-state PFA whose acceptance probability is identically equal to
        ``scalefactor``. If M is the result, then M satisfies
            M.prob(x) = scalefactor * self.prob(x)
        for all strings x, and M.size = self.size * 2.

        If ``self`` is a PFA and ``scalefactor`` is between 0 and 1, then
        ``self.scaled()`` is also a PFA. This is not generally true of the
        expression ``scalefactor*self``, which just multiplies ``self.initial``
        by ``scalefactor``.
        """
        return self.tensor_product(WeightedAutomaton.constant(self.ring(scalefactor),
                                                              self.alphabet))

    def direct_sum(self,other):
        """
        Return the direct sum of ``self`` with ``other``. This is the
        WeightedAutomaton whose initial and final state vectors are the
        concatenations of those of ``self`` and ``other``, respectively, and
        whose transition matrices are the block sums of those for ``self`` and
        ``other``. If ``self`` and ``other`` are both PFAs, their direct sum is
        again a PFA.

        If M = self.direct_sum(other), then M satisfies
            M.prob(x) = self.prob(x) + other.prob(x)
        for all strings x. Also M.size = self.size + other.size.

        The sets of variables of ``self`` and ``other`` will be combined in the
        output. If ``self`` and ``other`` have different rings, the output's
        ring will be either the larger of the two (if applicable), or SR.
        """
        if not isinstance(other,WeightedAutomaton):
            raise TypeError('you can only take the direct sum of two WeightedAutomata')
        if not self.alphabet == other.alphabet:
            raise TypeError('you can only take the direct sum of two WeightedAutomata over the same alphabet')
        # try to find the bigger of the two rings; if that fails, try shoving
        # everything into SR
        newring = self._biggerring(self.ring,other.ring)
        newinit = (self.initial.change_ring(newring)
                   .augment(other.initial.change_ring(newring)))
        newfinal = (self.final.change_ring(newring)
                    .stack(other.final.change_ring(newring)))
        newtrans = dict(zip(self.alphabet,
                            [self.transitions[a].block_sum(other.transitions[a])
                             for a in self.alphabet]))
        return WeightedAutomaton(newtrans,newinit,newfinal,ring=newring,
                                 variables=self.vars+other.vars)

    def elementwise_sum(self,other,selfweight=1,otherweight=1):
        """
        Return a WeightedAutomaton whose initial and final state vectors and
        transition matrices are equal to the sums of the respective quantities
        for ``self`` and ``other``. That is, if M is the returned
        WeightedAutomaton, we have
            M.initial = self.initial+other.initial; 
            M.final = self.final+other.final; and 
            M.transitions[a] = self.transitions[a] + other.transitions[a] 
        for every letter a.

        Optionally, you can set ``selfweight`` and/or ``otherweight`` to
        multiply every entry of the initial vector, final vector, and transition
        matrices of ``self`` by ``selfweight``, and likewise for ``otherweight``
        and ``other``.

        If some letter `b` is present in one of ``self``'s or ``other``'s
        alphabets but not in the other, then the returned WeightedAutomaton will
        have transition matrix for `b` equal to either ``self``'s or ``other``'s
        transition matrix for `b`, respectively.

        If ``self`` and ``other`` are both PFAs over the same alphabet, and if
        ``selfweight`` and ``otherweight`` are nonnegative numbers whose sum is
        1, then ``self.elementwise_sum(other,selfweight,otherweight)`` is also a
        PFA. In effect the result is a convex combination of ``self`` and
        ``other`` viewed as vectors in euclidean space.
        
        The sets of variables of ``self`` and ``other`` will be combined in the
        output. If ``self`` and ``other`` have different rings, the output's
        ring will be either the larger of the two (if applicable), or SR.
        """
        if not isinstance(other,WeightedAutomaton):
            raise TypeError('you can only take the elementwise sum of two WeightedAutomata')
        if not self.size == other.size:
            raise TypeError('elementwise sum only defined for WeightedAutomata of the same size')
        newring = self._biggerring(self.ring,other.ring)
        selfweight = newring(selfweight)
        otherweight = newring(otherweight)
        newdict = {}
        newinit = (selfweight*self.initial.change_ring(newring)
                   + otherweight*other.initial.change_ring(newring))
        newfinal = (selfweight*self.final.change_ring(newring)
                    + otherweight*other.final.change_ring(newring))
        for a in self.alphabet:
            newdict[a] = selfweight*self.transitions[a].change_ring(newring)
        for a in other.alphabet:
            # if a letter appears in both alphabets, add their matrices
            if a in self.alphabet:
                newdict[a] += (otherweight
                               *other.transitions[a].change_ring(newring))
            # if not, then just append other's transition matrix
            else:
                newdict[a] = (otherweight
                              *other.transitions[a].change_ring(newring))
        return WeightedAutomaton(newdict,newinit,newfinal,ring=newring,
                                 variables=self.vars+other.vars)

    def direct_product(self,other):
        """
        Return the direct product of ``self`` with ``other``. This is the
        WeightedAutomaton with ``self``'s initial state vector, ``other``'s
        final state vector, and whose transition matrix for letter a is
        equal to self.transitions[a] * other.transitions[a].

        Only defined between two WeightedAutomata over the same alphabet and
        with the same number of states. If ``self`` and ``other`` are both PFAs,
        their direct product is again a PFA of the same size.

        The sets of variables of ``self`` and ``other`` will be combined in the
        output. If ``self`` and ``other`` have different rings, the output's
        ring will be either the larger of the two (if applicable), or SR.
        """
        if not isinstance(other,WeightedAutomaton):
            raise TypeError('you can only take the direct product of two WeightedAutomata')
        if not (self.size == other.size and self.alphabet == other.alphabet):
            raise TypeError('direct product only defined for WeightedAutomata of the same size over the same alphabet')
        newring = self._biggerring(self.ring,other.ring)
        newdict = {}
        for a in self.alphabet:
            newdict[a] = (self.transitions[a]*other.transitions[a])
        return WeightedAutomaton(newdict, self.initial,
                                 other.final, ring=newring,
                                 variables=self.vars+other.vars)
    
    def tensor_product(self,other):
        """
        Return the tensor product of ``self`` with ``other``. This is the
        WeightedAutomaton whose initial and final state vectors and transition
        matrices are the tensor (Kronecker) products of the respective vectors
        and matrices for ``self`` and ``other``. If ``self`` and ``other`` are
        both PFAs, their tensor product is again a PFA.

        If M = self.tensor_product(other), then M satisfies
            M.prob(x) = self.prob(x) * other.prob(x)
        for all strings x. Also M.size = self.size * other.size.

        The sets of variables of ``self`` and ``other`` will be combined in the
        output. If ``self`` and ``other`` have different rings, the output's
        ring will be either the larger of the two (if applicable), or SR.
        """
        if not isinstance(other,WeightedAutomaton):
            raise TypeError('you can only take the tensor product of two WeightedAutomata')
        if not self.alphabet == other.alphabet:
            raise TypeError('you can only take the tensor product of two WeightedAutomata over the same alphabet')
        newring = self._biggerring(self.ring,other.ring)
        newinit = self.initial.tensor_product(other.initial, subdivide=False)
        newfinal = self.final.tensor_product(other.final, subdivide=False)
        newdict = {}
        for a in self.alphabet:
            newdict[a] = self.transitions[a].tensor_product(other.transitions[a], 
                                                            subdivide=False)
        return WeightedAutomaton(newdict,newinit,newfinal,ring=newring,
                                 variables=self.vars+other.vars)



################################################################################
################################################################################
################################################################################

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

    # TODO: kind of awkwardly written. Harmonize with highest_gap().
    def highest_prob(self,oflength=None):
        """
        Return the highest-probability strings among all strings in ``self``'s
        keys (if ``oflength``==None), or only among all strings of length
        ``oflength`` in ``self``'s keys (if ``oflength`` is set).
        Output: a list whose first entry is the highest probability and whose
        subsequent entries are all the strings of that probability.
        """
        # define usestrings to be the list of actual strings to compare
        if oflength == None:
            usestrings = list(self.keys())
        else:
            if not WeightedAutomaton._isint(oflength):
                raise TypeError(f'{oflength} is not a valid Sage or Python integer')
            if not oflength in self.lengths():
                raise ValueError(f'no strings of length {oflength} are present')
            usestrings = [k for k in self.keys() if len(k) == oflength]
        returnlist = []
        maxprob = max([self[w] for w in usestrings])
        returnlist.append(maxprob) # make the probability the first entry
        for w in usestrings:
            # append all strings with that probability
            if self[w] == maxprob:
                returnlist.append(w)
        return returnlist

    def highest_gap(self,oflength=None,numerical=False):
        """
        Return the gap between the two highest-probability strings among the
        keys of ``self``, or only among those of length ``oflength`` if the
        latter is specified.
        Output: a list in the form [gap, highest-prob string,
        second-highest-prob string].
        If ``numerical`` is set to True, the gap is listed as a decimal
        approximation (useful if you want to get a quick idea of the value).
        """
        # if the caller specified a length, it must be a valid length
        if oflength != None:
            if not WeightedAutomaton._isint(oflength):
                raise TypeError(f'{oflength} is not a valid Sage or Python integer')
            if not oflength in self.lengths():
                raise ValueError(f'no strings of length {oflength} are present')
            # compare only between strings of length oflength
            compare = self.slice(oflength)
        else:
            # compare between strings of all lengths represented in self.
            # (Obviously this will not actually produce a true 'gap' if more
            # than one length is present. Presumably the caller knows that.)
            compare = self
        # get 2 strings of highest prob, in order (using optimized library function)
        highest2 = heapq.nlargest(2, compare, key=compare.get)
        # prefix the list with the difference between their probs
        returncompare = compare[highest2[0]] - compare[highest2[1]]
        if numerical:
            returncompare = returncompare.n()
        return [returncompare] + highest2
    # TODO: it's weird to have highestgap() here but gap() in the PFA class. Maybe have both in both places, with the ones in PFA just being "compute the prob list, then run the PL func"?
