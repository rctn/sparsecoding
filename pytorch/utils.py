import numpy as np
import matplotlib.pyplot as plt


def makemontage(D,stride=4,color=False,colorf=None):
    """
    Put all dictionary elements into one large matrix image
    ---
    Parameters:
    D - scalar (n,n_basis)
        dictionary matrix, n==number of features
    stride - scalar (1,) default=4
        spacing between each dictionary element
    color - boolean (1,) defualt=false
        if color, dict columns are reshaped to (n,n,3)
    ---
    Returns:
    montage - scalar (stride*(n_basis-1)+griddim*pix,stride*(n_basis-1)+griddim*pix)
        normalized dictionary matrix
    """
    n,n_basis = D.shape
    if not color:
        pix = int(np.sqrt(n))
    else:
        pix = int(np.sqrt(n//3))

    # compute griddim (n elements in each row/col)
    squares = np.square(np.arange(0,24))
    griddim = squares[squares >= np.sqrt(n_basis)][0]
    montagepix = int(griddim*pix + stride*(griddim-1))
    if not color:
        montage = np.ones((montagepix,montagepix))
    else:
        montage = np.ones((montagepix,montagepix,3))
        Dcolor = D.reshape([pix**2,3,D.shape[1]])
        Dcolor = np.moveaxis(Dcolor,-1,0)
        Dcolor = Dcolor.reshape([-1,3])
    for i in range(n_basis):
        posy = i%griddim
        posx = i//griddim

        lr = stride*(posx)+pix*(posx)
        rr = lr+pix

        lc = stride*(posy)+pix*posy
        rc = lc+pix
        if not color:
            img = D[:,i].reshape([pix,pix])
            img = img/np.abs(img).max()
            montage[lr:rr,lc:rc] = img
        else:
            img = D[:,i].reshape([pix,pix,3])
            # img = (img - Dcolor.min(axis=0))/(Dcolor.max(axis=0) - Dcolor.min(axis=0))
            img = (img - img.min())/(img.max() - img.min())
            if colorf != None:
                img = colorf(img)
            montage[lr:rr,lc:rc,:] = img
    return montage
    
    
def plotmontage(model,stride=2,fig=None,ax=None,title='',size=8,dpi=100,color=False,colorf=None):
    """
    Plot monage of dictionary elements
    ---
    Parameters:
    model - class with member getnumpydict() that
        returns dictionary (n,n_basis)
    stride - scalar (1,) default=2
        spacing between each dictionary element
    fig - matplotlib figure default=None
    ax - matplotlib axis default=None
    title - string default=""
        title of plot
    size - scalar (1,) defualt=8
        size of plot (square)
    dpi - scalar (1,) defualt=100
        dots per inch resolution
    Returns:
    fig, ax - created or passes figure and axis handle
    """
    if fig == None or ax == None:
        fig,ax = plt.subplots(1,1,figsize=(size,size), dpi=dpi)
        
    montage = makemontage(model.getnumpydict(),stride=stride,color=color,colorf=colorf)
    ax.clear()
    ax.set_title(title)
    ax.imshow(montage,cmap='gray',vmin=-1,vmax=1,interpolation='nearest')
    ax.set_axis_off()
    fig.set_size_inches(size,size)
    fig.canvas.draw()
    return fig, ax
