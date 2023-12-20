import itertools
import numpy
import urllib,json
import pickle
import sys,trace
import multiprocessing

RRR = RealField(100);
eps = var('eps');

#stolen from https://ask.sagemath.org/question/8390/creating-an-array-of-variables/
class VariableGenerator(object):
     def __init__(self, prefix): 
         self.__prefix = prefix 
     @cached_method 
     def __getitem__(self, key): 
         return SR.var("%s%s"%(self.__prefix,key))

#object: what to dump; name: string without ".pickle"
def pickletofile(object,name):
    thefile = open(name+".pickle",'wb');
    pickle.dump(object,thefile);
    thefile.close();
    return

#filename: string without ".pickle"
def unpicklefile(filename):
    thefile = open(filename+".pickle",'rb');
    theobj = pickle.load(thefile);
    thefile.close();
    return theobj

def savewits():
    pickletofile(SC,"stoch_complexity");
    pickletofile(SW,"witnesses");
    return;

#NOTE: use dictionary NC for strings of length 2 through 10.
#expects sigma to be an array/list, like usual
def nfacomplexity(sigma):
    theurl = "http://math.hawaii.edu/~bjoern/complexity-api/?string="+''.join(str(ch) for ch in sigma);
    response = urllib.request.urlopen(theurl);
    data = json.loads(response.read());
    return int(data.get('complexity'))

#returns bit flip of a string (assumed binary)
def flip(sigma):
    l = "";
    for ch in sigma:
        l = l + str((1-int(ch)));
    return l;

#flips every string in an iterable, returns a Set
def flipset(strings):
    theset = [];
    for s in strings:
        theset.append(flip(s));
    return Set(theset)

#switches the order of a pair of matrices
def switchmats(matpair):
    return {'0': matpair['1'], '1': matpair['0']}

#utility function: convert everything in an iterable to a tuple.
#returns a tuple
def maketuple(iterable):
    newit = [];
    for i in iterable:
        newit.append(tuple(i));
    return tuple(newit);

#returns a string that is everything in atuple concatenated
def tuptostr(atuple):
    return ''.join([str(t) for t in atuple]);

#check if an object is an integer, either int or Sage integer
def isint(obj):
    return (isinstance(obj,sage.rings.integer.Integer) or isinstance(obj,int));

#return True iff NFA complexity of sigma is maximal
def isNFArand(sigma):
    assert sigma in NC.keys();
    return (NC[sigma] == floor(len(sigma)/2)+1);

#return True iff sigma uses at least nlett different letters
def usesletters(sigma,nlett):
    return (len(set(sigma)) >= nlett);

#returns "0.string" in base m
def madic_expansion(string, m):
    value = 0;
    counter = 0;
    while counter < len(string):
        assert int(string[counter]) < m; #should actually be an m-adic string
        value = value + int(string[counter])/(m^(counter+1));
        counter = counter + 1;
    return value;

#return image of string under a homomorphism homo
#homo should be a dictionary {letter:image} (both strings)
def homo_image(homo,string):
    image = "";
    for ch in string:
        assert ch in homo.keys(); #string should be in the domain of the homomorphism
        image = image + homo[ch];
    return image;

#expects start_states & accept_states to be lists
#transmatrices should be a dictionary. Its keys will be interpreted as letters of the input alphabet for the PFA
#returns acceptance probability of string sigma wrt the PFA given by transmatrices & start/accept_states
def accprob(transmatrices,sigma,start_states=None,accept_states=None):
    alph = list(transmatrices.keys());
    assert len(alph)>0;
    n = transmatrices[alph[0]].nrows();
    Id = identity_matrix(n);
    if start_states == None:
        v = Matrix(Id[0]);
    else:
        v = Matrix(start_states);
    if accept_states == None:
        f = Matrix(Id[n-1]).transpose();
    else:
        f = Matrix(accept_states).transpose();
    return accprob_vector(transmatrices,sigma,v,f);

#same, but expects start_states & accept_states to be matrices already
def accprob_vector(transmatrices,sigma,start_states,accept_states):
    A = start_states*probmatrix(transmatrices,sigma)*accept_states;
    return A[0][0];

#same, but expects a PFA given as a "witness" tuple consisting of (matrixdict, start_states, accept_states)
def accprob_witness(thePFA,sigma):
    return accprob_vector(thePFA[0],sigma,thePFA[1],thePFA[2]);
 
#return transition probability matrix for the string sigma
#transmatrices is a matrix dict giving the transition matrices for each letter
def probmatrix(transmatrices,sigma):
    alph = list(transmatrices.keys());
    A = identity_matrix(transmatrices[alph[0]].nrows());
    for ch in sigma:
        assert str(ch) in alph;
        A = A*transmatrices[ch];
    return A;

#lengths can be either an integer or a list
#return a list of pairs (string, accprob) for all strings of len in lengths.
#transmatrices should be a dictionary, {letter:transmatrix}
#start_states and accept_states can be lists or tuples
def list_probs(transmatrices,start_states=None,accept_states=None,lengths=range(1,5)):
    alph = list(transmatrices.keys());
    assert len(alph)>0;
    #if lengths is just a single integer, make it into a list
    if isint(lengths):
        lenlist = [lengths];
    else:
        lenlist = lengths;
    words = [];
    for i in lenlist:
        words = words + [tuptostr(it) for it in itertools.product(alph,repeat=i)];
    return list_probs_fromwords(transmatrices,start_states,accept_states,words);

#expects to be passed a "witness" tuple, as from SW
def list_probs_witness(witness,lengths=range(1,5)):
    return list_probs(witness[0],witness[1].list(),witness[2].list(),lengths)

#pass a list of strings to test, instead of a range
def list_probs_fromwords(transmatrices,start_states,accept_states,words):
    witness_tuple = (transmatrices,matrix(start_states),matrix(accept_states).transpose())
    return list_probs_fromwords_witness(witness_tuple,words)

#same, but expects to be passed a "witness" tuple
def list_probs_fromwords_witness(witness,words):
    problist = []; #list of strings with their acceptance probabilities
    for st in words:
        problist.append((st,accprob_witness(witness,st)));
    return problist;

#return list of max-prob strings of the specified length
def highest_prob_witness(thepfa,thelength):
    theproblist = list_probs_witness(thepfa,[thelength])
    return highest_prob_fromlist(theproblist,[thelength])

#expects a list in exactly the format returned by list_probs
#if lengths is a list (not None), only looks at strings of exactly those lengths
#if lengths is the string "all", returns highest-prob string of each length in problist
#if lengths is None, return the highest overall probability
def highest_prob_fromlist(problist,lengths=None):
    if lengths == None:
        highprob = max([prob[1] for prob in problist])
        return [st for st in problist if st[1] == highprob]
    elif lengths == "all":
        lengths = set([len(prob[0]) for prob in problist])
    highs = []; #list of highest strings of each length
    for l in lengths:
        sublist = [prob for prob in problist if len(prob[0]) == l]
        if len(sublist) == 0: continue; #allow there to be nothing of the given length
        highprob = max([prob[1] for prob in sublist])
        highs = highs + [st for st in sublist if st[1] == highprob]
    return highs
    
def sortproblist(problist):
    return sorted(problist,key=lambda x:x[1],reverse=True)
    
#check transition matrices for rows summing to 1
#P should be a dictionary of matrices
def checkmatrices(P):
    alph = list(transmatrices.keys());
    n = P[keys[0]].nrows();
    
    for i in keys:
        for j in range(n):
            s = sum(P[i][j]);
            if s != 1:
                if s > 1: relto1 = " > 1";
                elif s < 1: relto1 = " < 1";
                print("P"+str(i)+" row "+str(j+1)+" sum is "+str(RRR(s))+relto1);

#given list of row vectors and alphabet, returns all possible PFAs (matrix dicts) using those rows.
#vecs = list of vectors (which should be tuples!)
#if makerational=True, make all the matrices in the end over QQ. Don't bother otherwise
def bruteforcevecs(vecs,alph=range(2),permute=True,info=False,makerational=True):
    nstates = len(vecs[0]);
    mats = Tuples(vecs,nstates);
    #now convert every single element of mats to a tuple (immutable)
    tmats = maketuple(mats);
    if info: print(str(len(tmats))+" matrices");
    #tmats = pool of possible matrices to pull from. Now build all
    #possible dictionaries with alph as keys.
    numletters = len(alph);
    if permute:
        mats2 = list(Tuples(tmats,numletters));
    else:
        mats2 = list(itertools.combinations(tmats,numletters));
    if info: print(str(len(mats2))+" matrix tuples");
    Pnew = []; #will be list of matrix dicts
    for P in mats2:
        newdict = {}; #we'll add to this a matrix for each letter, in order of appearance in P
        if makerational:
            for i in range(len(alph)):
                newdict.update({str(alph[i]):Matrix(QQ,P[i])});
        else:
            for i in range(len(alph)):
                newdict.update({str(alph[i]):Matrix(P[i])});
        Pnew.append(newdict);
    return Pnew;

#return list of dictionaries of nstates x nstates matrices, over
#the alphabet alph (which should be a list or tuple).
#(1/step) will be the increment of transition probabilities
#if permute=False, don't include (a) all ways to order a
#tuple of matrices; (b) tuples where any two mats are the same.
#if info=True, print sizes of stuff along the way
#if prec>0, make the entries have precision prec. If prec=0, they're in QQ
#if maxdenom=True, discard matrix combos that have maximum denominator < step
def bruteforce(nstates,step,alph=range(2),permute=True,info=False,prec=0,maxdenom=False):
    assert(prec >= 0);
    linit = Compositions(nstates+step,length=nstates);
    if info: print(str(len(linit))+" partitions");
    #lrows = [ [0]*nstates ];
    lrows = []; #not including rows of all 0s anymore
    if prec > 0:
        field = RealField(prec);
    else:
        field = QQ;
    for t in linit:
        r = []; #do arithmetic on t by hand
        for i in t:
            r.append(field(i-1)/step);
        lrows.append(tuple(r));
    trows = tuple(lrows);
    if info: print(str(len(trows))+" rows");
    theresult = bruteforcevecs(trows,alph,permute,info,(prec==0));
    #filter out stuff if maxdenom=True. If step is a prime number, then no need to do this (it'll only fail to remove deterministic ones)
    if maxdenom and not is_prime(step):
        thenewresult = [];
        for m in theresult:
            if matdenom(m) == step:
                thenewresult.append(m); #and otherwise don't
        theresult = thenewresult;
    return theresult;

#do bruteforce() algo using only the given list of transition probabilities as entries
#if oneminus=True: if a probability p is present but not 1-p, then 1-p will automatically be added
#if useeps=True, the matrices will just be over SR. Automatically overrides makerational to False.
#if makerational=True, convert all given probs to rationals
def bruteforceprobs(nstates,probs,alph=range(2),permute=True,info=False,useeps=False,makerational=True,oneminus=True):
    #for p in probs:
    #    if p<0 or p>1:
    #        probs.remove(p);
    if makerational and not useeps:
        pro_list = [];
        for p in probs:
            pro_list.append(QQ(p));
        if oneminus:
            for p in pro_list:
                if pro_list.count(1-p) == 0:
                    pro_list.append(QQ(1-p));
        pro = tuple(pro_list);
    else:
        pro_list = copy(probs);
        if oneminus:
            for p in pro_list:
                if pro_list.count(1-p) == 0:
                    pro_list.append(1-p);
        pro = tuple(pro_list);
    if info: print(str(len(pro))+" possible entries");
    lrowsall = Tuples(pro,nstates).list();
    lrows = [];
    for r in lrowsall:
        s = sum(r);
        if s == 1: lrows.append(r);
    if info: print(str(len(lrows))+" rows");
    trows = maketuple(lrows);
    if not useeps:
        return bruteforcevecs(trows,alph,permute,info,makerational);
    else:
        return bruteforcevecs(trows,alph,permute,info,False);

#returns a list of lists [wit,(word,prob)] where wit gives word highest acceptance probability prob among strings of the same length
#matslist = list of dictionaries of matrices as returned by bruteforce()
#lengths = lengths of strings to try (list), accept_states = vector (list)
#this one only uses a single start/accept vector for the whole list
def finduniques(matslist,lengths,start_states=None,accept_states=None):
    alph = list(matslist[0].keys())
    n = matslist[0][alph[0]].nrows()
    uniques = []
    if start_states == None:
        start_states = list(identity_matrix(n)[0])
    if accept_states == None:
        accept_states = list(identity_matrix(n)[n-1])
    start_states = matrix(start_states)
    accept_states = matrix(accept_states).transpose()
    return finduniques_witness( [ (w,start_states,accept_states) for w in matslist], lengths)
#    words = [];
#    if isint(lengths):
#        lenlist = [lengths];
#    else:
#        lenlist = lengths;
#    for l in lenlist:
#        words = words + [tuptostr(s) for s in itertools.product(alph,repeat=l)];
#    for P in matslist:
#        probs = list_probs_fromwords(P,start_states,accept_states,words);
#        high = highest_prob(probs);
#        if len(high) == 1:
#            uniques.append([P,high[0]]);
#    return uniques;

#same, but expects thewits to be a list of "witness tuples" (matdict, initial state vector, final state vector)
#this one allows each witness to come with different initial/final vectors
def finduniques_witness(thewits,lengths):
    alph = list(thewits[0][0].keys())
    uniques = []
    if isint(lengths):
        lenlist = [lengths]
    else:
        lenlist = lengths
    words = []
    for l in lenlist:
        words = words + [tuptostr(s) for s in itertools.product(alph,repeat=l)]
    for w in thewits:
        probs = list_probs_fromwords_witness(w,words)
        high = highest_prob_fromlist(probs)
        if len(high) == 1:
            uniques.append([w,high[0]])
    return uniques

#updates PFA complexities and adds new witnesses to SW
#uniqlist should be in the format output by finduniques()
#ASSUMES ALL MATRICES IN uniqlist ARE OF THE SAME SIZE!!!!!
#specify the _same_ list of accept states for all of them.
#return number of new witnesses added
def addwits(uniqlist,start_states=None,accept_states=None):
    alph = list(uniqlist[0][0].keys());
    n = uniqlist[0][0][alph[0]].nrows();
    count = 0;
    if start_states == None:
        v = Matrix(identity_matrix(n)[0]);
    else:
        v = Matrix(start_states);
    if accept_states == None:
        f = Matrix(identity_matrix(n)[n-1]).transpose();
    else:
        f = Matrix(accept_states).transpose();
    for u in uniqlist:
        s = u[1][0];
        count = count + addonewit(u[0],s,start_states,accept_states);
    return count;

#like addwits(), but only adds if there isn't a witness yet for
#each particular string
#uniqlist should be in the format output by finduniques()
#ASSUMES ALL MATRICES IN uniqlist ARE OF THE SAME SIZE!!!!!
#specify the _same_ list of accept states for all of them.
#return number of new witnesses added
def addwitsnew(uniqlist,start_states=None,accept_states=None):
    alph = list(uniqlist[0][0].keys());
    n = uniqlist[0][0][str(alph[0])].nrows();
    count = 0;
    for u in uniqlist:
        s = u[1][0];
        if s in SW.keys() and len(SW[s]) > 0 and s in SC.keys():
            continue;
        count = count + addonewit(u[0],s,start_states,accept_states);
    return count;

#adds thewit (a matrix dict) as a witness for sigma to SW
#also updates SC if appropriate
#only does it if this witnesses minimal complexity
#return 0 if nothing was added or updated;
#       1 if only a new witness;
#       2 if new witness and complexity updated.
def addonewit(thewit,sigma,start_states=None,accept_states=None):
    n = thewit[str(sigma[0])].nrows();
    ret = 0;
    if start_states == None:
        v = Matrix(identity_matrix(n)[0]);
    else:
        assert n == len(start_states);
        v = Matrix(start_states);
    if accept_states == None:
        f = Matrix(identity_matrix(n)[n-1]).transpose();
    else:
        assert n == len(accept_states);
        f = Matrix(accept_states).transpose();
    return addonewit_vec(thewit,sigma,v,f);

#same as addonewit(), but start/accept_states should now already be a
#matrix in the correct form
def addonewit_vec(thewit,sigma,start_states,accept_states):
    n = thewit[str(sigma[0])].nrows();
    ret = 0;
    if not sigma in SC.keys() or SC[sigma] > n:
        SC[sigma] = n;
        ret = ret + 1;
    if SC[sigma] == n:
        if not sigma in SW.keys(): SW[sigma] = [];
        newwit = (thewit,start_states,accept_states);
        if not newwit in SW[sigma]:
            SW[sigma].append(newwit);
            ret = ret + 1;
    return ret;

#same again, but expecting a witness triple
def addonewit_wit(thewit,sigma):
    return addonewit_vec(thewit[0],sigma,thewit[1],thewit[2]);

#if thepfa (in witness form) gives sigma highest prob among strings of the same length, return the list of all such probs
# when this happens the first entry in the list will always be (sigma,prob of sigma)
#if not, return None
#optional: fromstrings is an iterable of strings from which to draw from. if empty, will be filled with all strings of same length
def ishighestprob(thepfa,sigma,alph=range(2),fromstrings=None):
    n = len(sigma)
    if fromstrings == None: # if no set provided, fill out with all strings of length |sigma|
        fromstrings = [tuptostr(it) for it in itertools.product(alph,repeat=n)]
        # DON'T delete sigma because i suspect it does some memory optimization if you run it a billion times w/ same set
    problist = [] # list of all strings of this length with their probs, which might not be finished
    sigprob = accprob_witness(thepfa,sigma)
    problist.append((sigma,sigprob))
    for s in fromstrings:
        if s == sigma: continue # don't compare sigma with sigma!
        compareprob = accprob_witness(thepfa,s)
        if compareprob >= sigprob:
            return None
        problist.append((s,compareprob)) # fall through to add the prob to our growing list
    return problist

#----------------------------------------------------------------------------------
#---------------BEGIN FUNCTIONS NOT FULLY VETTED FOR NEW PFA FORMAT----------------
#----------------------------------------------------------------------------------

#among PFAs given by matdicts, returns the first found which
#has sigma as its highest-prob word among strings of len(sigma).
#assumes all matrices have the same size.
#if stop>0, find the first stop witnesses (if possible).
#if stop=1, the single witness (if found) is not returned inside a list.
#if stop=0, find all witnesses, if possible.
def findwitness(matdicts,sigma,start_states=None,accept_states=None,stop=1):
    n = matdicts[0][str(sigma[0])].nrows();
    if start_states == None:
        v = Matrix(identity_matrix(n)[0]);
    else:
        v = Matrix(start_states);
    if accept_states == None:
        f = matrix(identity_matrix(n)[n-1]).transpose();
    else:
        f = matrix(accept_states).transpose();
    words = [tuptostr(s) for s in itertools.product(list(matdicts[0].keys()),repeat=len(sigma))];
    words.remove(sigma); #these are the words whose acc. probs we're comparing against sigma's
    
    found = 0;
    wits = [];
    for P in matdicts:
        sigprob = accprob_vector(P,sigma,v,f);
        if sigprob == 0:
            continue;
        iscandidate = True;
        for s in words:
            testprob = accprob_vector(P,s,v,f);
            if testprob >= sigprob:
                iscandidate = False;
                break;
        if iscandidate:
            if stop != 1:
                wits.append((P,v,f));
            elif stop == 1:
                return P; #the traditional behavior: don't put the single witness into a list
            found = found + 1;
            if stop > 0 and found >= stop:
                return wits;
    if len(wits) == 0:
        return None;
    else:
        return wits;

#expects a list in the format given by finduniques
#returns the first pair of matrices in uniqlist witnessing the
#PFA complexity of sigma
def firstwitness(uniqlist,sigma):
    return [u[0] for u in uniqlist if u[1][0][0] == sigma][0]

#expects a dictionary like SC or NC
#returns Set of keys in compdict with value=complexity
#restricts to length strlen if strlen>0; o/w no restr on length
#if less=True, returns keys with value<=complexity
def lookup(compdict,complexity,strlen=0,less=False):
    if strlen == 0 and not less:
        return Set([k for k in compdict.keys() if compdict[k] == complexity]);
    elif strlen > 0 and not less:
        return Set([k for k in compdict.keys() if compdict[k] == complexity and len(k) == strlen]);
    elif strlen == 0 and less:
        return Set([k for k in compdict.keys() if compdict[k] <= complexity]);
    elif strlen > 0 and less:
        return Set([k for k in compdict.keys() if compdict[k] <= complexity and len(k) == strlen]);


#return largest denominator in a dict of rational matrices
def matdenom(matdict):
    #TODO: make sure the matrices are actually rational
    themax = 1;
    for i in matdict.keys():
        for r in matdict[i]: #rows
            for e in r: #element
                d = QQ(e).denominator();
                themax = lcm(d,themax);
    return themax;

#return first witness for PFA complexity of sigma in SW with
#matdenom <= denom
def firstwitdenom(sigma,denom):
    if not SW.has_key(sigma) or len(SW[sigma]) == 0:
        return None;
    for mats in SW[sigma]:
        if matdenom(mats[0]) <= denom:
            return mats;
    return None;

#return list of witnesses for PFA complexity of sigma in SW
#with matdenom = denom
def witsdenom(sigma,denom):
    if not SW.has_key(sigma) or len(SW[sigma]) == 0:
        return None;
    thelist = [];
    for mats in SW[sigma]:
        if matdenom(mats[0]) == denom:
            thelist.append(mats);
    if thelist == []:
        return None;
    return thelist;

#return lowest denominator out of all witnesses for sigma in SW
def mindenom(sigma):
    if not SW.has_key(sigma) or len(SW[sigma]) == 0:
        return None;
    return min([matdenom(w[0]) for w in SW[sigma]]);

#return highest denominator out of all witnesses for sigma in SW
def maxdenom(sigma):
    if not SW.has_key(sigma) or len(SW[sigma]) == 0:
        return None;
    return max([matdenom(w[0]) for w in SW[sigma]]);

#determine if matpair represents an actual aut (as in Rabin)
def isactualaut(matpair):
    for i in range(2):
        for r in matpair[i]: #rows
            for e in r: #entries
                if e == 0:
                    return False;
    return True;

#substitute epses (tuple of epsilons, one tuple per matrix,
#one epsilon per row) into matpairs for the variable eps
def matsub(matpairs,epses):
    assert len(epses[0]) == len(epses[1]) == matpairs[0].nrows();
    newmat = [ [], [] ];
    for i in range(2):
        for e in range(len(epses[i])):
            newmat[i].append(matpairs[i][e].substitute(eps=epses[i][e]));
    return [Matrix(QQ,newmat[0]), Matrix(QQ,newmat[1])];

#just returns the digraph of the aut described by matpair
def autgraph(matpair):
    n = matpair[0].nrows();
    gr = DiGraph(n,loops=True,multiedges=True);
    for l in range(2): #letter seen
        for i in range(n): #transition from state i...
            for j in range(n): #to state j
                if matpair[l][i][j] > 0:
                    gr.add_edge(i,j,label=str(l)+" ("+str(matpair[l][i][j])+")");
    return gr;

#plots the specified aut from SW
#its=# of iterations for plotting algo
def autplotwit(sigma,ind,its=2):
    matpair = SW[sigma][ind][0];
    f = SW[sigma][ind][1];
    n = f.nrows();
    gr = autgraph(matpair);
    acc = [i for i in range(n) if f[i][0] == 1];
    notacc = Set(range(n)).difference(Set(acc)).list();
    return gr.plot(edge_labels=True,vertex_size=400,layout='spring',iterations=its,vertex_colors={'white': notacc,'yellow': acc});

#expects a matrix dict defining the automaton
def autplot(matdict,accept_states=None,its=2):
    n = matpair[matdict.keys()[0]].nrows();
    if accept_states == None:
        f = Matrix(identity_matrix(n)[n-1]).transpose();
        acc = [n-1];
    else:
        f = Matrix(accept_states).transpose();
        acc = [i for i in range(n) if accept_states[i] == 1];
    gr = autgraph(matpair);
    notacc = Set(range(n)).difference(Set(acc)).list();
    return gr.plot(edge_labels=True,vertex_size=400,layout='spring',iterations=its,vertex_colors={'white': notacc,'yellow': acc});

#expects a digraph (probably the output of autgraph())
def autgraphplot(autg,accept_states=None,its=2):
    n = autg.order();
    if accept_states == None:
        f = Matrix(identity_matrix(n)[n-1]).transpose();
        acc = [n-1];
    else:
        f = Matrix(accept_states).transpose();
        acc = [i for i in range(n) if accept_states[i] == 1];
    notacc = Set(range(n)).difference(Set(acc)).list();
    return autg.plot(edge_labels=True,vertex_size=400,layout='spring',iterations=its,vertex_colors={'white': notacc,'yellow': acc});

#----------------------------------------------------------------------------------
#---------------END FUNCTIONS NOT FULLY VETTED FOR NEW PFA FORMAT------------------
#----------------------------------------------------------------------------------

#given PFA (as matrix dictionary and init/acc states) and string sigma, 
#print a list of all accepting paths for sigma (as a sequence of 
#states) and the probability accumulated along each path
#accept_states must be a list/tuple
#for simplicity, for now only works for a single start state
def pathlist(matdict,sigma,accept_states=None):
    alph = list(matdict.keys());
    n = matdict[alph[0]].nrows(); # #states
    l = len(sigma)+1; #length of path
    #if start_states == None:
    #    v = Matrix(identity_matrix(n)[0]);
    #else:
    #    v = Matrix(start_states);
    if accept_states == None:
        accept_states = matrix(identity_matrix(n)[n-1]).transpose().list();
    #list of all possible sequences of states of the correct length
    stateseqs = [];
    for i in range(n):
        if accept_states[i] == 1: #if ith state is accepting
            for tup in Tuples(range(n),l-2):
                tupli = list(tup);
                tupli.insert(0,0); #begin with state 0 (before making any state transition)
                tupli.append(i); #end with state i, which we know is accepting
                stateseqs.append(tupli);
    #list of all such paths with their probs
    seqsnprobs = [];
    for aseq in stateseqs:
        pathprob = 1;
        for k in range(l-1):
            pathprob = pathprob*matdict[sigma[k]][aseq[k]][aseq[k+1]];
        seqsnprobs.append((aseq,pathprob));
    theoutput = "";
    for s in seqsnprobs:
        if s[1] > 0:
            addstr = "path "+str(s[0])+" has prob "+str(s[1])+'\n';
            #print(addstr,flush=True)
            theoutput = theoutput + addstr;
    return theoutput;

#expects theaut to be a "witness" triple. Still (for now) ignores the actual initial state distribution
def pathlist_witness(theaut,sigma):
    return pathlist(theaut[0],sigma,theaut[2].list());

#given a list of strings, a pool of matrix dicts, and start/accept states,
#try to find a witness for each string in the pool.
#will print any successes, and stop after trying maxattempts strings. If maxattemps=0, try all of them.
#won't attempt to find witnesses if one already exists.
#if dontbother=True, don't bother to look for stuff if we already have witnesses. In this case, also don't add them to SW
def tryfindwitness(strings,pool,start_states=None,accept_states=None,maxattempts=50,dontbother=True):
    c = 0;
    for s in strings:
        if (s in SW.keys() and len(SW[s]) > 0) or NC[s] == 1:
            if dontbother: continue;
        c = c + 1;
        if c < 0:
            continue;
        if c > maxattempts and maxattempts != 0:
            break;
        #sys.stdout.write("%d\r" % (c) );
        print(c,s);
        v = start_states;
        f = accept_states;
        wits = findwitness(pool,s,v,f);
        if wits != None:
            print(s,wits,v,f);
            if dontbother: print(addonewit(wits,s,v,f));
            print("");
            print(flip(s),switchmats(wits),v,f);
            if dontbother: print(addonewit(switchmats(wits),flip(s),v,f));
            print("");
        else:
            print("no witness found");
            print("");
            
#returns an m-adic PFA, a la Salomaa/Turakainen. It is a tuple: (matrices, initial, accepting) (as in SW)
#input: homo, a dict giving a homomorphism. Keys are generators of the language, values are their images (all strings)
def madic(homo):
    alph = list(homo.keys());
    m = max([int(a) for a in alph])+1;
    themats = {}; #matrix dict to generate
    for letter in alph:
        phi = madic_expansion(homo[letter],m);
        malpha = m^(-len(homo[letter]));
        themats.update({letter: Matrix(QQ, [[malpha, 1-malpha-phi, phi],
                                           [0,1,0],
                                           [0,0,1]])});
    return (themats, Matrix([1,0,0]), Matrix([0,0,1]).transpose());

#given a PFA in "witness" form, return a new PFA whose transition matrices are multiplied by the matrices
#in mask. That is, matrices[letter] becomes matrices[letter]*mask[letter]. (Right multiplication by default,
#but set left=True if you want left multiplication.)
#Does NOT check the types of anything or the sizes of matrices
def multiplyPFA(thePFA,mask,left=False):
    newmats = {};
    for letter in thePFA[0].keys():
        assert letter in mask.keys();
        if left:
            newmats.update({letter:mask[letter]*thePFA[0][letter]});
        else:
            newmats.update({letter:thePFA[0][letter]*mask[letter]});
    return (newmats,thePFA[1],thePFA[2]);

#return a single random rational stochastic vector of length length and denominator denom
def random_stoch_vector(length,denom):
    assert length>0 and denom>0
    the_ints = [randint(0,denom)] #start of list of random ints in progressively narrower range
    for i in range(1,length-1): #i=1 through length-2, which will result in the second-to-last entry of the_ints
        the_ints.append(randint(0,denom-sum(the_ints[0:i]))) #we want the total to sum to no more than denom
    the_ints.append(denom-sum(the_ints)) #make the sum = denom
    return vector(the_ints)/denom

#return a random stochastic matrix over QQ of size nxn and denominator denom
def random_stoch(n,denom):
    rows = [[]]*n;
    for i in range(n):
        rows[i] = list(random_stoch_vector(n,denom))
    return matrix(QQ,rows)
    
#return a random stochastic matrix over QQ of size nxn. It generates the entries as real numbers first
#precision = #bits of precision (default is RR)
def random_stoch_real(n,precision=53):
    P = random_matrix(RealField(precision),n,n).change_ring(QQ)
    #the following two lines stolen from https://drvinceknight.blogspot.com/2013/10/pigeon-holes-markov-chains-and-sagemath.html
    P = [[abs(k) for k in row] for row in P]  # This ensures all our numbers are positive
    P = matrix(QQ,[[k / sum(row) for k in row] for row in P]) # This ensures that our rows all sum to 1
    return P

#returns a random PFA with nstates states, denominator denom, over alphabet alph (so far only does a single final state)
def random_PFA(nstates,denom,alph=['0','1']):
    matdict = {}
    for a in alph:
        matdict[a] = random_stoch(nstates,denom)
    return (matdict, random_stoch_vector(nstates,denom), matrix(identity_matrix(nstates)[0]).transpose() )

#returns the gap of the specified PFA (in witness form!) wrt sigma. This is the minimum difference of rho(sigma)
#and rho(tau) over all strings with |tau|=|sigma| and tau!=sigma.
#Note that if thepfa doesn't witness an upper bound for A_P(sigma), this value will be negative.
def probgap(thepfa,sigma):
    n = len(sigma)
    #pull only the probabilities of each other word of length n
    thelist = set([p[1] for p in list_probs_witness(thepfa,n) if p[0] != sigma])
    sigprob = accprob_witness(thepfa,sigma) #there's a smarter way to do this
    differencelist = [sigprob - otherprob for otherprob in thelist];
    return min(differencelist) #smallest such value is the gap, by definition

#input: matrix dict and two lists/tuples of states
#output: a tuple in "witness form"
def towit(transmatrices,start_states,accept_states):
    return (transmatrices,matrix(start_states),matrix(accept_states).transpose())

# return true iff thepfa has a dead state. Expects it in witness form and over QQ.
def checkdeadstate(thepfa):
    firstletter = list(thepfa[0].keys())[0]
    nstates = len(thepfa[0][firstletter][0])
    for i in range(nstates): # for each row
        # if this is an accepting state, it doesn't count as dead (imo)
        # (also, need the extra [0] because Sage converts each entry of a vector into its own tuple
        if list(thepfa[2])[i][0] == 1: continue
        dead_evidence = 0 # number of row coincidences: if = # of keys, you have a dead state
        for sigma in thepfa[0].keys(): # for each transition matrix
#            rational_version = thepfa[0][sigma].change_ring(QQ) # REALLY expensive
            if thepfa[0][sigma][i] == identity_matrix(nstates)[i]: # if the ith row agrees with the ith row of the identity matrix
                dead_evidence = dead_evidence + 1 # one more piece of evidence towards a dead verdict
        if dead_evidence == len(thepfa[0].keys()): # if there are as many pieces of evidence as there are matrices
            return True
    return False

#---------------------------------------------------------------------------
#------------------IFS-RELATED FUNCTIONS------------------------------------
#---------------------------------------------------------------------------

#input: list of functions of one variable; output: their composition in reverse order
#that is, if your list is [f0,f1,f2], returns f0\circ f1\circ f2.
#they're assumed to be over the same ring, use sensibly
def composelist(funclist):
    l = len(funclist);
    if l == 1:
        return funclist[0]; #no need to do anything if it's a single function
    outfun = funclist[l-1];
    for n in range(1,l):
        outfun = compose(funclist[l-n-1],outfun);
    return outfun;

#expects thepfa in "witness form", i.e., tuple (matrix dict, initial vector, final vector)
#returns tuple (f0,f1,x0)
#only supports 2-state binary PFAs
def pfa2ifs(thepfa):
    assert thepfa[0]['0'].nrows() == 2;
    vecaslist = list(thepfa[2].transpose()[0]); #breaking down cases of final state vector
    if vecaslist == [0,0]:
        return (lambda x:0, lambda x:0, 0);
    elif vecaslist == [1,1]:
        return (lambda x:1, lambda x:1, 1);
    #remaining 2 cases fall through
    elif vecaslist == [1,0]: #don't need to switch around anything
        mymat0 = thepfa[0]['0'];
        mymat1 = thepfa[0]['1'];
        mystart = thepfa[1].list()[0];
    elif vecaslist == [0,1]: #switch all rows and columns
        mymat0 = thepfa[0]['0'].with_swapped_rows(0,1).with_swapped_columns(0,1);
        mymat1 = thepfa[0]['1'].with_swapped_rows(0,1).with_swapped_columns(0,1);
        mystart = thepfa[1].list()[1];
    f0(x) = mymat0[1][0] + (mymat0[0][0]-mymat0[1][0])*x; #a+bx, where b = (top left entry) - a
    f1(x) = mymat1[1][0] + (mymat1[0][0]-mymat1[1][0])*x;
    return (lambda x:f0(x),lambda x:f1(x),mystart);

#supports only 2-state PFAs, so expects ifs to be in the usual tuple form (f0,f1,x0)
#returns a 2-state PFA in "witness form" (matrix dict, start states, accept states), and
#start states = accept states = (1,0) (at least for now).
def ifs2pfa(ifs):
    assert len(ifs) == 3;
    f0 = ifs[0].function(x); #to be on the safe side
    f1 = ifs[1].function(x);
    x0 = ifs[2];
    #yes this is very stupidly written, I don't feel like being clever
    a = QQ(f0.coefficient(x,0));
    b = QQ(f0.coefficient(x,1));
    c = QQ(f1.coefficient(x,0));
    d = QQ(f1.coefficient(x,1));
    return ({'0': matrix([[a+b,1-a-b],[a,1-a]]), '1': matrix([[c+d,1-c-d],[c,1-c]])}, matrix([x0,1-x0]), matrix([1,0]).transpose());

#denom = equivalent of "step" from regular bruteforce(), xstart = starting value of x
#return list of triples (f0(x),f1(x),xstart) with all possible rational coeffs w/denominator denom
def bruteforce_ifs(denom,xstart,alphsize=2):  
    #generate all possible pairs (a,b) for a single matrix
    ablist = [];
    for i in range(denom):
        ai = i/denom; #our current "a" value
        for j in range(-denom,denom+1):
            bj = j/denom;
            if 0 <= ai + bj <= 1:
                ablist.append((ai,bj));
    #list of pairs of (a,b)s, corresponding to dicts of matrices
    matlist = Tuples(ablist,alphsize);
    #populate the list of IFSs: a triple of (f0,f1,xstart)
    ifslist = [];
    for m in matlist:
        ifslist.append( (m[0][0]+m[0][1]*x, m[1][0]+m[1][1]*x, xstart) );
    return ifslist;

#ifs = triple (f0,f1,xstart); sigma is the string to test
#returns the acceptance probability of sigma by the PFA corresponding to ifs
def accprob_ifs(ifs,sigma):
    #list of functions to pass to composelist()
    tocompose = [];
    for ch in sigma:
        tocompose.insert(0,ifs[int(ch)]); #insert in reverse order, effectively
    return composelist(tocompose)(x=ifs[2]);

#ifs = as above; lengths = as in list_probs()
#returns a list in exactly the same format as list_probs() would
def list_probs_ifs(ifs,lengths=range(1,5)):
    alph=[0,1]; #for now
    #if lengths is just a single integer, make it into a list
    if isint(lengths):
        lenlist = [lengths];
    else:
        lenlist = lengths;
    words = [];
    for i in lenlist:
        words = words + [tuptostr(it) for it in itertools.product(alph,repeat=i)];
    problist = [];
    for st in words:
        problist.append( (st, accprob_ifs(ifs,st)) );
    return problist;
#I know the above should be refactored in a similar way to the original list_probs(), maybe later


#run immediately when restarting
SC = unpicklefile("stoch_complexity");
NC = unpicklefile("nfa_complexity");
SW = unpicklefile("witnesses");


S2 = {}; #set of binary strings of reasonable length
S2[0] = Set([]);
for i in range(2,15):
    S2[i] = Set([tuptostr(it) for it in itertools.product([0,1],repeat=i)]);
    S2[0] = S2[0].union(S2[i]);
    
S3 = {}; #ternary strings that use all 3 letters
S3all = {}; #all ternary strings
S3[0] = Set([]);
S3all[0] = Set([]);
for i in range(2,11):
    initial = [tuptostr(it) for it in itertools.product([0,1,2],repeat=i)];
    final = [];
    for s in initial:
        if usesletters(s,3):
            final.append(s);
    S3[i] = Set(final);
    S3[0] = S3[0].union(S3[i]);
    S3all[i] = Set(initial);
    S3all[0] = S3all[0].union(S3all[i]);
