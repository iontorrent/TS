#include <stdlib.h>
#include <stdio.h>


int main(int argc, char** argv){

    int *a, *b;
    char **m;
    int i,j,j_1,k,n,r;

    if (argc<3) {
        printf("usage: ./multi_erik k item_1 ... item_n\n");
        return 0;
    }
    n = atoi(argv[1]);
    r = argc - 2;

    m = malloc(r * sizeof(char*));
    a = malloc(n * sizeof(int));
    b = malloc(n * sizeof(int));

    for (i=2;i<argc;i++)
        m[i-1] = argv[i];

    for (i=1;i<=n;i++) {
        a[i] = 1; b[i] = r;
    }

    j=n;
    while(1){
        // emit multiset combination
        for(i=1;i<=n;i++)
            printf("%s ", m[a[i]]);
        printf("\n");
        j=n;
        while(a[j]==b[j])j--;
        if (j<=0) break;
        j_1=j;
        while(j_1<=n){
            a[j_1]=a[j_1]+1;
            k=j_1;
            while(k<n) {
                a[k+1]=a[k];
                k++;
            }
            k++;
            j_1=k;
        }
    }

    return 0;
}

