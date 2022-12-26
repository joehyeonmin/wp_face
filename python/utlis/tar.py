import tarfile

fname = 'vggface2_test.tar.gz'  
ap = tarfile.open(fname)     
ap.extractall()
ap.close()
