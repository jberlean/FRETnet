import numpy as np
from scipy import linalg, optimize,sparse
from scipy.linalg import interpolative
from PIL import Image


# I accidentally 1-indexed everything and haven't fixed it yet - sorry :(

def get_position(i,j,n):
    """returns position in vector for k_(ij)
        n: number of variables
        returns it 0-indexed, not 1-indexed but assumes i and j are 1-indexed
    """
    small = min(i,j)
    big = max(i,j)
    return (small-1)*n - ((small-1)*small)//2 + (big-small) - 1



def make_matrix_from_pattern(patterns):
    """goal is to make a matrix usable in get_weights based on the provided m patterns
        patterns is an m x n numpy array
        The output matrix is mn x n(n-1)/2 (assume m < (n-1)/2)
        order for the variables is all the k_(ij) sorted by (i,j) for i<j
    """
    # get the number and length of the patterns and start the matrix up
    m,n = patterns.shape
    M = np.zeros((m*n,(n*(n-1))//2))
    # M = sparse.lil_matrix((m*n,(n*(n-1))//2))
    # go through each pattern
    for p in range(1,m+1):
        for i in range(1,n+1):
            for j in range(i+1,n+1):
                # figure out the position of (i,j)
                across = get_position(i,j,n)
                difference = patterns[p-1,j-1] -patterns[p-1,i-1]
                
                # pick row i
                down = n*(p-1) + i - 1
                M[down,across] = difference
                
                # plug in the value for (j,i)
                # pick row j
                down = n*(p-1) + j -1
                M[down,across] = -difference

              
    return M

def get_weights(patterns):
    """patterns: m x n numpy array of binary patterns
        each pattern forms a row
        there are m patterns of length n
        we must have m = (n+1)/2 for this value to work
    returns: an n(n+1)/2 x 1 numpy array of weights
        the value for k_(ij) for i<j is located at (i-1)(n-i/2)+(j-i) = (i-1)n - (i-1)i/2 + (j-i)
            the second one might be easier to work with since the term with /2 is always whole
            this number is also one indexed so you need to subtract 1 for python indexing
        the value for k_(is) is located at (n(n-1)/2)+i
    """
    # get the matrix we need to solve this on and the inputs as a function of the patterns
    M = make_matrix_from_pattern(patterns)
    print(M.shape,np.linalg.matrix_rank(M))
    ks = inp1(patterns)
    kis = inp2(patterns)
    # flatten the input patterns into a single vector
    p = np.array([np.matrix.flatten(patterns)]).T

    # get the right side of the equation
    b = -1*ks + np.multiply(ks,p) + np.multiply(kis,p)
    # turn into a vector to be compatible
    b = b[:,0]
    
    # get the lsq solution to Mx=b
    optimization = optimize.lsq_linear(M,b)
    result = optimization.x
    print("least squares")
    print(result)

    # get an orthonormal basis for the nullspace of M
    null = linalg.null_space(M)
    print("nullspace")
    print(null)
 
    # figure out what we're going to be adding to the result
    extra = np.matrix.flatten(np.sum(null,axis=1,keepdims=True))
    # return the final weights
    return np.add(result,extra)



def getA2(weights,n):
    """weights: an n(n-1)/2 x 1 array of the weights (output of get_weights)
        n: the value of n (number of nodes/length of the pattern)
    returns: an n x n array of the constant matrix A2
    """    
    # start A2
    A2 = np.zeros((n,n))
    # do the non-diagonal first, will end up adding things for the diagonal
    for i in range(n):
        for j in range(n):
            if i != j:
                pos = get_position(i+1,j+1,n)

                A2[i,j] = weights[pos]
    # now add in the diagonal terms
    for i in range(n):
        total = np.sum(A2[:,i])
        A2[i,i] = -total
    return A2

def getA1(ks,kis):
    """ks and kis are the vectors of ksi and kis values 
        return A1 based on these
    """
    # get a 1D array
    negks = -1*(ks[:,0]+kis[:,0])
    # return a diagonal matrix with the above values on the diagonal
    return np.diag(negks)

def run_network(ks,kis,weights,n):
    """ks,kis: inputs of the ksi and kis values 
            these are the inputs and can be found from the expected p values using 
            inp1 and inp2 values respectively
        weights: output of get weights, should have all the k_(ij) values
        n: the length of the patterns
    returns the steady state p* values
    """
    # get the overall value for A
    A = getA1(ks,kis)+getA2(weights,n)
    # get the right side
    b = -1*ks.T
    # return solution
    return linalg.solve(A,b.T)

def inp1(patterns,default=1):
    """gets the ksi vector
        ksi = 1 if pi is 1
        ksi = number of 1s/length of pattern if pi is 0
    """
    m,n = patterns.shape 
    out = []
    for i in range(m):
        num_ones = sum(patterns[i,:])
        out += [default if p == 1 else num_ones/n for p in patterns[i,:]]
    return np.array([out]).T


def inp2(patterns,default=1):
    """gets the kis vector
        kis is 1 if pi is 0
        kis is number of 0s/length of pattern if pi is 1
    """
    m,n = patterns.shape 
    out = []
    for i in range(m):
        num_zeros = n-sum(patterns[i,:])
        out += [default if p == 0 else num_zeros/n for p in patterns[i,:]]
    return np.array([out]).T


def test_all_patterns(patterns,weights,threshold=True,size=None,saving_names=None,prin=False):
    """takes in the patterns and the weights and can print out the result vs input for each pattern
        and/or can save them back as images
        if Threshold is True, everything above .5 becomes 1, everything below becomes 0
        size: the size of the image you want to save if you're saving the image
    """
    print("testing all patterns")
    m,n = patterns.shape 
    for i in range(m):
        # get pattern i
        pattern = patterns[i:i+1,:]
        # get inputs for i
        ks = inp1(pattern)
        kis = inp2(pattern)
        # run the network with the inputs
        result = run_network(ks,kis,weights,n)
        print(min(result),max(result))

        # threshold if needed
        if threshold:
            for j in range(n):
                result[j,0] = 0 if result[j,0] < .5 else 1
        
        # print if needed
        if prin:
            print("input vs guessed")
            print(np.vstack((pattern,result.T)).T)

        # save if needed
        if saving_names:
            save_image(result,size,saving_names[i])


def get_pixels(names):
    """names: list of m names of images
    returns an m x n matrix where n is the number of pixels per image
        each row is a list of pixels in the image
    also returns a tuple that is the size of the image
    """
    # get the image size and the values
    size = Image.open(names[0]).size
    values = list(Image.open(names[0]).getdata())
    # get the first entry and set value to 0 or 1
    pixels = [0 if pix[0] < 255/2 else 1 for pix in values]
    # start up the new array
    patterns = np.array([pixels])
    # add all the other ones to the matrix
    for name in names[1:]:
        values = list(Image.open(name).getdata())
        pixels = [0 if pix[0] < 255/2 else 1 for pix in values]
        patterns = np.vstack((patterns,pixels))
    return patterns,size

def save_image(pixels,size,filename):
    """pixels: a vector or list of pixels
        size: tuple wiht size of image to save 
        (product of length and width should be the length of pixels)
        filename: the name you should save the file under
    """
    # get a list of real pixels from the input
    pixels = [p*255 for p in pixels]
    # start a new image and put your pixels in it, then save
    im = Image.new("L",size)
    im.putdata(pixels)
    im.save(filename)


if __name__ == "__main__":
    training_patterns = np.array([ [1,0,0]
                                    ])
    testing_patterns = np.array([ [1,0,0],
                                   [0,1,0],
                                   [1,1,0],
                                   [0,1,1],
                                   [0,0,0],
                                   [1,1,1]
                                   ])
    # # (make_matrix_from_pattern(training_patterns))
    weights = get_weights(training_patterns)
    print("weights",weights)
    test_all_patterns(testing_patterns,weights,threshold=False,prin=True)
    # print(linalg.inv(np.array([[-11/3,1,1],
    #                            [1,-10/3,1],
    #                            [1,1,-10/3]
    #                             ])))

    # training_prefix = "medium_patterns/testing/"
    # testing_prefix = "medium_patterns/testing/"
    # saving_prefix = "medium_patterns/results/"
    # start_patterns = ["tree.png","smiley.png","triangle.png","stripes.png"]
    # test_patterns = ["tree.png","smiley.png","triangle.png","tree_corrupted.png","tree_corrupted2.png","smiley_corrupted.png","smiley_corrupted2.png",
    #                 "random.png","triangle_corrupted.png","triangle_inverted.png","smiley_inverted.png","tree_corrupted2_inverted.png","stripes.png",
    #                 "stripes_corrupted.png","stripes_inverted.png"]

    # training_names = [training_prefix + name for name in start_patterns]
    # testing_names = [testing_prefix + name for name in test_patterns]
    # saving_names = [saving_prefix + name for name in test_patterns]

    # training_patterns,size = get_pixels(training_names)
    # weights = get_weights(training_patterns)
    # print("got weights!")

    # testing_patterns,size = get_pixels(testing_names)
    # test_all_patterns(testing_patterns,weights,size=size,saving_names=saving_names,threshold=True)



