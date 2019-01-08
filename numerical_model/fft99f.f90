! (C) Copyright 2009-2016 ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!
! PURPOSE      PERFORMS MULTIPLE FAST FOURIER TRANSFORMS.  THIS PACKAGE
!           WILL PERFORM A NUMBER OF SIMULTANEOUS REAL/HALF-COMPLEX
!           PERIODIC FOURIER TRANSFORMS OR CORRESPONDING INVERSE
!           TRANSFORMS, I.E.  GIVEN A SET OF REAL DATA VECTORS, THE
!           PACKAGE RETURNS A SET OF 'HALF-COMPLEX' FOURIER
!           COEFFICIENT VECTORS, OR VICE VERSA.  THE LENGTH OF THE
!           TRANSFORMS MUST BE AN EVEN NUMBER GREATER THAN 4 THAT HAS
!           NO OTHER FACTORS EXCEPT POSSIBLY POWERS OF 2, 3, AND 5.
!           THIS IS AN ALL FORTRAN VERSION OF THE CRAYLIB PACKAGE
!           THAT IS MOSTLY WRITTEN IN CAL.
!
!           THE PACKAGE FFT99F CONTAINS SEVERAL USER-LEVEL ROUTINES:
!
!         SUBROUTINE SET99
!             AN INITIALIZATION ROUTINE THAT MUST BE CALLED ONCE
!             BEFORE A SEQUENCE OF CALLS TO THE FFT ROUTINES
!             (PROVIDED THAT N IS NOT CHANGED).
!
!         SUBROUTINES FFT99 AND FFT991
!             TWO FFT ROUTINES THAT RETURN SLIGHTLY DIFFERENT
!             ARRANGEMENTS OF THE DATA IN GRIDPOINT SPACE.
!
!
! ACCESS       THIS FORTRAN VERSION MAY BE ACCESSED WITH
!
!                *FORTRAN,P=XLIB,SN=FFT99F
!
!           TO ACCESS THE CRAY OBJECT CODE, CALLING THE USER ENTRY
!           POINTS FROM A CRAY PROGRAM IS SUFFICIENT.  THE SOURCE
!           FORTRAN AND CAL CODE FOR THE CRAYLIB VERSION MAY BE
!           ACCESSED USING
!
!                FETCH P=CRAYLIB,SN=FFT99
!                FETCH P=CRAYLIB,SN=CAL99
!
! USAGE        LET N BE OF THE FORM 2**P * 3**Q * 5**R, WHERE P .GE. 1,
!           Q .GE. 0, AND R .GE. 0.  THEN A TYPICAL SEQUENCE OF
!           CALLS TO TRANSFORM A GIVEN SET OF REAL VECTORS OF LENGTH
!           N TO A SET OF 'HALF-COMPLEX' FOURIER COEFFICIENT VECTORS
!           OF LENGTH N IS
!
!                DIMENSION IFAX(13),TRIGS(3*N/2+1),A(M*(N+2)),
!               +          WORK(M*(N+1))
!
!                CALL SET99 (TRIGS, IFAX, N)
!                CALL FFT99 (A,WORK,TRIGS,IFAX,INC,JUMP,N,M,ISIGN)
!
!           SEE THE INDIVIDUAL WRITE-UPS FOR SET99, FFT99, AND
!           FFT991 BELOW, FOR A DETAILED DESCRIPTION OF THE
!           ARGUMENTS.
!
! HISTORY      THE PACKAGE WAS WRITTEN BY CLIVE TEMPERTON AT ECMWF IN
!           NOVEMBER, 1978.  IT WAS MODIFIED, DOCUMENTED, AND TESTED
!           FOR NCAR BY RUSS REW IN SEPTEMBER, 1980.
!
!-----------------------------------------------------------------------
!
! SUBROUTINE SET99 (TRIGS, IFAX, N)
!
! PURPOSE      A SET-UP ROUTINE FOR FFT99 AND FFT991.  IT NEED ONLY BE
!           CALLED ONCE BEFORE A SEQUENCE OF CALLS TO THE FFT
!           ROUTINES (PROVIDED THAT N IS NOT CHANGED).
!
! ARGUMENT     IFAX(13),TRIGS(3*N/2+1)
! DIMENSIONS
!
! ARGUMENTS
!
! ON INPUT     TRIGS
!            A FLOATING POINT ARRAY OF DIMENSION 3*N/2 IF N/2 IS
!            EVEN, OR 3*N/2+1 IF N/2 IS ODD.
!
!           IFAX
!            AN INTEGER ARRAY.  THE NUMBER OF ELEMENTS ACTUALLY USED
!            WILL DEPEND ON THE FACTORIZATION OF N.  DIMENSIONING
!            IFAX FOR 13 SUFFICES FOR ALL N LESS THAN A MILLION.
!
!           N
!            AN EVEN NUMBER GREATER THAN 4 THAT HAS NO PRIME FACTOR
!            GREATER THAN 5.  N IS THE LENGTH OF THE TRANSFORMS (SEE
!            THE DOCUMENTATION FOR FFT99 AND FFT991 FOR THE
!            DEFINITIONS OF THE TRANSFORMS).
!
! ON OUTPUT    IFAX
!            CONTAINS THE FACTORIZATION OF N/2.  IFAX(1) IS THE
!            NUMBER OF FACTORS, AND THE FACTORS THEMSELVES ARE STORED
!            IN IFAX(2),IFAX(3),...  IF SET99 IS CALLED WITH N ODD,
!            OR IF N HAS ANY PRIME FACTORS GREATER THAN 5, IFAX(1)
!            IS SET TO -99.
!
!           TRIGS
!            AN ARRAY OF TRIGNOMENTRIC FUNCTION VALUES SUBSEQUENTLY
!            USED BY THE FFT ROUTINES.
!
!-----------------------------------------------------------------------
!
! SUBROUTINE FFT991 (A,WORK,TRIGS,IFAX,INC,JUMP,N,M,ISIGN)
!                    AND
! SUBROUTINE FFT99 (A,WORK,TRIGS,IFAX,INC,JUMP,N,M,ISIGN)
!
! PURPOSE      PERFORM A NUMBER OF SIMULTANEOUS REAL/HALF-COMPLEX
!           PERIODIC FOURIER TRANSFORMS OR CORRESPONDING INVERSE
!           TRANSFORMS, USING ORDINARY SPATIAL ORDER OF GRIDPOINT
!           VALUES (FFT991) OR EXPLICIT CYCLIC CONTINUITY IN THE
!           GRIDPOINT VALUES (FFT99).  GIVEN A SET
!           OF REAL DATA VECTORS, THE PACKAGE RETURNS A SET OF
!           'HALF-COMPLEX' FOURIER COEFFICIENT VECTORS, OR VICE
!           VERSA.  THE LENGTH OF THE TRANSFORMS MUST BE AN EVEN
!           NUMBER THAT HAS NO OTHER FACTORS EXCEPT POSSIBLY POWERS
!           OF 2, 3, AND 5.  THESE VERSION OF FFT991 AND FFT99 ARE
!           OPTIMIZED FOR USE ON THE CRAY-1.
!
! ARGUMENT     A(M*(N+2)), WORK(M*(N+1)), TRIGS(3*N/2+1), IFAX(13)
! DIMENSIONS
!
! ARGUMENTS
!
! ON INPUT     A
!            AN ARRAY OF LENGTH M*(N+2) CONTAINING THE INPUT DATA
!            OR COEFFICIENT VECTORS.  THIS ARRAY IS OVERWRITTEN BY
!            THE RESULTS.
!
!           WORK
!            A WORK ARRAY OF DIMENSION M*(N+1)
!
!           TRIGS
!            AN ARRAY SET UP BY SET99, WHICH MUST BE CALLED FIRST.
!
!           IFAX
!            AN ARRAY SET UP BY SET99, WHICH MUST BE CALLED FIRST.
!
!           INC
!            THE INCREMENT (IN WORDS) BETWEEN SUCCESSIVE ELEMENTS OF
!            EACH DATA OR COEFFICIENT VECTOR (E.G.  INC=1 FOR
!            CONSECUTIVELY STORED DATA).
!
!           JUMP
!            THE INCREMENT (IN WORDS) BETWEEN THE FIRST ELEMENTS OF
!            SUCCESSIVE DATA OR COEFFICIENT VECTORS.  ON THE CRAY-1,
!            TRY TO ARRANGE DATA SO THAT JUMP IS NOT A MULTIPLE OF 8
!            (TO AVOID MEMORY BANK CONFLICTS).  FOR CLARIFICATION OF
!            INC AND JUMP, SEE THE EXAMPLES BELOW.
!
!           N
!            THE LENGTH OF EACH TRANSFORM (SEE DEFINITION OF
!            TRANSFORMS, BELOW).
!
!           M
!            THE NUMBER OF TRANSFORMS TO BE DONE SIMULTANEOUSLY.
!
!           ISIGN
!            = +1 FOR A TRANSFORM FROM FOURIER COEFFICIENTS TO
!                 GRIDPOINT VALUES.
!            = -1 FOR A TRANSFORM FROM GRIDPOINT VALUES TO FOURIER
!                 COEFFICIENTS.
!
! ON OUTPUT    A
!            IF ISIGN = +1, AND M COEFFICIENT VECTORS ARE SUPPLIED
!            EACH CONTAINING THE SEQUENCE:
!
!            A(0),B(0),A(1),B(1),...,A(N/2),B(N/2)  (N+2 VALUES)
!
!            THEN THE RESULT CONSISTS OF M DATA VECTORS EACH
!            CONTAINING THE CORRESPONDING N+2 GRIDPOINT VALUES:
!
!            FOR FFT991, X(0), X(1), X(2),...,X(N-1),0,0.
!            FOR FFT99, X(N-1),X(0),X(1),X(2),...,X(N-1),X(0).
!                (EXPLICIT CYCLIC CONTINUITY)
!
!            WHEN ISIGN = +1, THE TRANSFORM IS DEFINED BY:
!              X(J)=SUM(K=0,...,N-1)(C(K)*EXP(2*I*J*K*PI/N))
!              WHERE C(K)=A(K)+I*B(K) AND C(N-K)=A(K)-I*B(K)
!              AND I=SQRT (-1)
!
!            IF ISIGN = -1, AND M DATA VECTORS ARE SUPPLIED EACH
!            CONTAINING A SEQUENCE OF GRIDPOINT VALUES X(J) AS
!            DEFINED ABOVE, THEN THE RESULT CONSISTS OF M VECTORS
!            EACH CONTAINING THE CORRESPONDING FOURIER COFFICIENTS
!            A(K), B(K), 0 .LE. K .LE N/2.
!
!            WHEN ISIGN = -1, THE INVERSE TRANSFORM IS DEFINED BY:
!              C(K)=(1/N)*SUM(J=0,...,N-1)(X(J)*EXP(-2*I*J*K*PI/N))
!              WHERE C(K)=A(K)+I*B(K) AND I=SQRT(-1)
!
!            A CALL WITH ISIGN=+1 FOLLOWED BY A CALL WITH ISIGN=-1
!            (OR VICE VERSA) RETURNS THE ORIGINAL DATA.
!
!            NOTE: THE FACT THAT THE GRIDPOINT VALUES X(J) ARE REAL
!            IMPLIES THAT B(0)=B(N/2)=0.  FOR A CALL WITH ISIGN=+1,
!            IT IS NOT ACTUALLY NECESSARY TO SUPPLY THESE ZEROS.
!
! EXAMPLES      GIVEN 19 DATA VECTORS EACH OF LENGTH 64 (+2 FOR EXPLICIT
!            CYCLIC CONTINUITY), COMPUTE THE CORRESPONDING VECTORS OF
!            FOURIER COEFFICIENTS.  THE DATA MAY, FOR EXAMPLE, BE
!            ARRANGED LIKE THIS:
!
! FIRST DATA   A(1)=    . . .                A(66)=             A(70)
! VECTOR       X(63) X(0) X(1) X(2) ... X(63) X(0)  (4 EMPTY LOCATIONS)
!
! SECOND DATA  A(71)=   . . .                                  A(140)
! VECTOR       X(63) X(0) X(1) X(2) ... X(63) X(0)  (4 EMPTY LOCATIONS)
!
!            AND SO ON.  HERE INC=1, JUMP=70, N=64, M=19, ISIGN=-1,
!            AND FFT99 SHOULD BE USED (BECAUSE OF THE EXPLICIT CYCLIC
!            CONTINUITY).
!
!            ALTERNATIVELY THE DATA MAY BE ARRANGED LIKE THIS:
!
!             FIRST         SECOND                          LAST
!             DATA          DATA                            DATA
!             VECTOR        VECTOR                          VECTOR
!
!              A(1)=         A(2)=                           A(19)=
!
!              X(63)         X(63)       . . .               X(63)
!     A(20)=   X(0)          X(0)        . . .               X(0)
!     A(39)=   X(1)          X(1)        . . .               X(1)
!               .             .                               .
!               .             .                               .
!               .             .                               .
!
!            IN WHICH CASE WE HAVE INC=19, JUMP=1, AND THE REMAINING
!            PARAMETERS ARE THE SAME AS BEFORE.  IN EITHER CASE, EACH
!            COEFFICIENT VECTOR OVERWRITES THE CORRESPONDING INPUT
!            DATA VECTOR.
!
!-----------------------------------------------------------------------
SUBROUTINE fft99a(a,work,trigs,inc,jump,n,lot)
  implicit none
  integer, intent(in) :: inc, jump, n, lot
  real(8), intent(in) :: a(n), trigs(n)
  real(8),intent(out) :: work(n)

  real(8) :: c, s
  integer :: ia, iabase, ib, ibbase, ink, ja, jabase, jb, jbbase, &
           & k, l, nh, nx
!
!  SUBROUTINE FFT99A - PREPROCESSING STEP FOR FFT99, ISIGN=+1
!  (SPECTRAL TO GRIDPOINT TRANSFORM)
!
  nh=n/2
  nx=n+1
  ink=inc+inc
!
!  A(0) AND A(N/2)
  ia=1
  ib=n*inc+1
  ja=1
  jb=2
!DIR$ IVDEP
  DO l=1,lot
    work(ja)=a(ia)+a(ib)
    work(jb)=a(ia)-a(ib)
    ia=ia+jump
    ib=ib+jump
    ja=ja+nx
    jb=jb+nx
  END DO
!
!  REMAINING WAVENUMBERS
  iabase=2*inc+1
  ibbase=(n-2)*inc+1
  jabase=3
  jbbase=n-1
!
  DO k=3,nh,2
    ia=iabase
    ib=ibbase
    ja=jabase
    jb=jbbase
    c=trigs(n+k)
    s=trigs(n+k+1)
!DIR$ IVDEP
    DO l=1,lot
      work(ja)=(a(ia)+a(ib))-                                           &
       &  (s*(a(ia)-a(ib))+c*(a(ia+inc)+a(ib+inc)))
      work(jb)=(a(ia)+a(ib))+                                           &
       &  (s*(a(ia)-a(ib))+c*(a(ia+inc)+a(ib+inc)))
      work(ja+1)=(c*(a(ia)-a(ib))-s*(a(ia+inc)+a(ib+inc)))+             &
       &  (a(ia+inc)-a(ib+inc))
      work(jb+1)=(c*(a(ia)-a(ib))-s*(a(ia+inc)+a(ib+inc)))-             &
       &  (a(ia+inc)-a(ib+inc))
      ia=ia+jump
      ib=ib+jump
      ja=ja+nx
      jb=jb+nx
    END DO
    iabase=iabase+ink
    ibbase=ibbase-ink
    jabase=jabase+2
    jbbase=jbbase-2
  END DO
!
  IF (iabase /= ibbase) GO TO 50
!  WAVENUMBER N/4 (IF IT EXISTS)
  ia=iabase
  ja=jabase
!DIR$ IVDEP
  DO l=1,lot
    work(ja)=2.0_8*a(ia)
    work(ja+1)=-2.0_8*a(ia+inc)
    ia=ia+jump
    ja=ja+nx
  END DO
!
  50 CONTINUE
  RETURN
END SUBROUTINE fft99a

SUBROUTINE fft99b(work,a,trigs,inc,jump,n,lot)
  implicit none
  integer, intent(in) :: inc, jump, n, lot
  real(8),intent(in) :: work(n), trigs(n)
  real(8),intent(out) :: a(n)

  integer :: ia, iabase, ib, ibbase, ink, ja, jabase, jb, jbbase, &
           & k, l, nh, nx
  real(8) :: c, s, scale
!
!  SUBROUTINE FFT99B - POSTPROCESSING STEP FOR FFT99, ISIGN=-1
!  (GRIDPOINT TO SPECTRAL TRANSFORM)
!
  nh=n/2
  nx=n+1
  ink=inc+inc
!
!  A(0) AND A(N/2)
  scale=1.0_8/REAL(n,8)
  ia=1
  ib=2
  ja=1
  jb=n*inc+1
!DIR$ IVDEP
  DO l=1,lot
    a(ja)=scale*(work(ia)+work(ib))
    a(jb)=scale*(work(ia)-work(ib))
    a(ja+inc)=0.0_8
    a(jb+inc)=0.0_8
    ia=ia+nx
    ib=ib+nx
    ja=ja+jump
    jb=jb+jump
  END DO
!
!  REMAINING WAVENUMBERS
  scale=0.5_8*scale
  iabase=3
  ibbase=n-1
  jabase=2*inc+1
  jbbase=(n-2)*inc+1
!
  DO k=3,nh,2
    ia=iabase
    ib=ibbase
    ja=jabase
    jb=jbbase
    c=trigs(n+k)
    s=trigs(n+k+1)
!DIR$ IVDEP
    DO l=1,lot
      a(ja)=scale*((work(ia)+work(ib))                                  &
       &  +(c*(work(ia+1)+work(ib+1))+s*(work(ia)-work(ib))))
      a(jb)=scale*((work(ia)+work(ib))                                  &
       &  -(c*(work(ia+1)+work(ib+1))+s*(work(ia)-work(ib))))
      a(ja+inc)=scale*((c*(work(ia)-work(ib))-s*(work(ia+1)+work(ib+1))) &
       &  +(work(ib+1)-work(ia+1)))
      a(jb+inc)=scale*((c*(work(ia)-work(ib))-s*(work(ia+1)+work(ib+1))) &
       &  -(work(ib+1)-work(ia+1)))
      ia=ia+nx
      ib=ib+nx
      ja=ja+jump
      jb=jb+jump
    END DO
    iabase=iabase+2
    ibbase=ibbase-2
    jabase=jabase+ink
    jbbase=jbbase-ink
  END DO
!
  IF (iabase /= ibbase) GO TO 50
!  WAVENUMBER N/4 (IF IT EXISTS)
  ia=iabase
  ja=jabase
  scale=2.0_8*scale
!DIR$ IVDEP
  DO l=1,lot
    a(ja)=scale*work(ia)
    a(ja+inc)=-scale*work(ia+1)
    ia=ia+nx
    ja=ja+jump
  END DO
!
  50 CONTINUE
  RETURN
END SUBROUTINE fft99b

SUBROUTINE fft991(a,work,trigs,ifax,inc,jump,n,lot,ISIGN)
  implicit none
  integer, intent(in) :: inc, jump, n, lot, isign
  integer, intent(in) :: ifax(1)
  real(8), intent(in) :: trigs(n)
  real(8), intent(inout) :: a(n)
  real(8), intent(out) :: work(n)

  integer :: i, ia, ib, ibase, igo, ink, j, jbase, k, l, la, m, nfax, &
           & nh, nx
!
!  SUBROUTINE "FFT991" - MULTIPLE REAL/HALF-COMPLEX PERIODIC
!  FAST FOURIER TRANSFORM
!
!  SAME AS FFT99 EXCEPT THAT ORDERING OF DATA CORRESPONDS TO
!  THAT IN MRFFT2
!
!  PROCEDURE USED TO CONVERT TO HALF-LENGTH COMPLEX TRANSFORM
!  IS GIVEN BY COOLEY, LEWIS AND WELCH (J. SOUND VIB., VOL. 12
!  (1970), 315-337)
!
!  A IS THE ARRAY CONTAINING INPUT AND OUTPUT DATA
!  WORK IS AN AREA OF SIZE (N+1)*LOT
!  TRIGS IS A PREVIOUSLY PREPARED LIST OF TRIG FUNCTION VALUES
!  IFAX IS A PREVIOUSLY PREPARED LIST OF FACTORS OF N/2
!  INC IS THE INCREMENT WITHIN EACH DATA 'VECTOR'
!      (E.G. INC=1 FOR CONSECUTIVELY STORED DATA)
!  JUMP IS THE INCREMENT BETWEEN THE START OF EACH DATA VECTOR
!  N IS THE LENGTH OF THE DATA VECTORS
!  LOT IS THE NUMBER OF DATA VECTORS
!  ISIGN = +1 FOR TRANSFORM FROM SPECTRAL TO GRIDPOINT
!        = -1 FOR TRANSFORM FROM GRIDPOINT TO SPECTRAL
!
!  ORDERING OF COEFFICIENTS:
!      A(0),B(0),A(1),B(1),A(2),B(2),...,A(N/2),B(N/2)
!      WHERE B(0)=B(N/2)=0; (N+2) LOCATIONS REQUIRED
!
!  ORDERING OF DATA:
!      X(0),X(1),X(2),...,X(N-1)
!
!  VECTORIZATION IS ACHIEVED ON CRAY BY DOING THE TRANSFORMS IN
!  PARALLEL
!
!  *** N.B. N IS ASSUMED TO BE AN EVEN NUMBER
!
!  DEFINITION OF TRANSFORMS:
!  -------------------------
!
!  ISIGN=+1: X(J)=SUM(K=0,...,N-1)(C(K)*EXP(2*I*J*K*PI/N))
!      WHERE C(K)=A(K)+I*B(K) AND C(N-K)=A(K)-I*B(K)
!
!  ISIGN=-1: A(K)=(1/N)*SUM(J=0,...,N-1)(X(J)*COS(2*J*K*PI/N))
!            B(K)=-(1/N)*SUM(J=0,...,N-1)(X(J)*SIN(2*J*K*PI/N))
!
!
!
  nfax=ifax(1)
  nx=n+1
  nh=n/2
  ink=inc+inc
  IF (ISIGN == +1) GO TO 30
!
!  IF NECESSARY, TRANSFER DATA TO WORK AREA
  igo=50
  IF (MOD(nfax,2) == 1) GO TO 40
  ibase=1
  jbase=1
  DO l=1,lot
    i=ibase
    j=jbase
!DIR$ IVDEP
    DO m=1,n
      work(j)=a(i)
      i=i+inc
      j=j+1
    END DO
    ibase=ibase+jump
    jbase=jbase+nx
  END DO
!
  igo=60
  GO TO 40
!
!  PREPROCESSING (ISIGN=+1)
!  ------------------------
!
  30 CONTINUE
  CALL fft99a (a,work,trigs,inc,jump,n,lot)
  igo=60
!
!  COMPLEX TRANSFORM
!  -----------------
!
  40 CONTINUE
  ia=1
  la=1
  DO k=1,nfax
    IF (igo == 60) GO TO 60
!    50 CONTINUE
    CALL vpassm (a(ia),a(ia+inc),work(1),work(2),trigs,                  &
        ink,2,jump,nx,lot,nh,ifax(k+1),la)
    igo=60
    GO TO 70
    60 CONTINUE
    CALL vpassm (work(1),work(2),a(ia),a(ia+inc),trigs,                  &
     &  2,ink,nx,jump,lot,nh,ifax(k+1),la)
    igo=50
    70 CONTINUE
    la=la*ifax(k+1)
  END DO
!
  IF (ISIGN == -1) GO TO 130
!
!  IF NECESSARY, TRANSFER DATA FROM WORK AREA
  IF (MOD(nfax,2) == 1) GO TO 110
  ibase=1
  jbase=1
  DO l=1,lot
    i=ibase
    j=jbase
!DIR$ IVDEP
    DO m=1,n
      a(j)=work(i)
      i=i+1
      j=j+inc
    END DO
    ibase=ibase+nx
    jbase=jbase+jump
  END DO
!
!  FILL IN ZEROS AT END
  110 CONTINUE
  ib=n*inc+1
!DIR$ IVDEP
  DO l=1,lot
    a(ib)=0.0_8
    a(ib+inc)=0.0_8
    ib=ib+jump
  END DO
  GO TO 140
!
!  POSTPROCESSING (ISIGN=-1):
!  --------------------------
!
  130 CONTINUE
  CALL fft99b (work,a,trigs,inc,jump,n,lot)
!
  140 CONTINUE
  RETURN
END SUBROUTINE fft991

SUBROUTINE set99 (trigs, ifax, n)
  implicit none
  integer, intent(in) :: n
  integer, intent(out) :: ifax(13)
  real(8), intent(out) :: trigs(3*n/2+1)

  integer :: i, mode
!
! MODE 3 IS USED FOR REAL/HALF-COMPLEX TRANSFORMS.  IT IS POSSIBLE
! TO DO COMPLEX/COMPLEX TRANSFORMS WITH OTHER VALUES OF MODE, BUT
! DOCUMENTATION OF THE DETAILS WERE NOT AVAILABLE WHEN THIS ROUTINE
! WAS WRITTEN.
!
  DATA mode /3/
  CALL fax (ifax, n, mode)
  i = ifax(1)
  IF (ifax(i+1) > 5 .OR. n <= 4) ifax(1) = -99
  IF (ifax(1) <= 0 ) THEN
    CALL abor1_ftn('abor1_ftn called from SET99: INVALID N')
  END IF
  CALL fftrig (trigs, n, mode)
  RETURN
END SUBROUTINE set99

SUBROUTINE fax(ifax,n,mode)
  DIMENSION ifax(10)
  nn=n
  IF (IABS(mode) == 1) GO TO 10
  IF (IABS(mode) == 8) GO TO 10
  nn=n/2
  IF ((nn+nn) == n) GO TO 10
  ifax(1)=-99
  RETURN
  10 k=1
!  TEST FOR FACTORS OF 4
  20 IF (MOD(nn,4) /= 0) GO TO 30
  k=k+1
  ifax(k)=4
  nn=nn/4
  IF (nn == 1) GO TO 80
  GO TO 20
!  TEST FOR EXTRA FACTOR OF 2
  30 IF (MOD(nn,2) /= 0) GO TO 40
  k=k+1
  ifax(k)=2
  nn=nn/2
  IF (nn == 1) GO TO 80
!  TEST FOR FACTORS OF 3
  40 IF (MOD(nn,3) /= 0) GO TO 50
  k=k+1
  ifax(k)=3
  nn=nn/3
  IF (nn == 1) GO TO 80
  GO TO 40
!  NOW FIND REMAINING FACTORS
  50 l=5
  inc=2
!  INC ALTERNATELY TAKES ON VALUES 2 AND 4
  60 IF (MOD(nn,l) /= 0) GO TO 70
  k=k+1
  ifax(k)=l
  nn=nn/l
  IF (nn == 1) GO TO 80
  GO TO 60
  70 l=l+inc
  inc=6-inc
  GO TO 60
  80 ifax(1)=k-1
!  IFAX(1) CONTAINS NUMBER OF FACTORS
  nfax=ifax(1)
!  SORT FACTORS INTO ASCENDING ORDER
  IF (nfax == 1) GO TO 110
  DO ii=2,nfax
    istop=nfax+2-ii
    DO i=2,istop
      IF (ifax(i+1) >= ifax(i)) CYCLE
      item=ifax(i)
      ifax(i)=ifax(i+1)
      ifax(i+1)=item
    END DO
  END DO
  110 CONTINUE
  RETURN
END SUBROUTINE fax

SUBROUTINE fftrig(trigs,n,mode)
  implicit none
  integer, intent(in) :: n, mode
  real(8), intent(out) :: trigs(1)

  integer :: i, imode, l, la, nh, nn
  real(8) :: angle, del, pi

  pi=2.0_8*ASIN(1.0_8)
  imode=IABS(mode)
  nn=n
  IF (imode > 1.AND.imode < 6) nn=n/2
  del=(pi+pi)/REAl(nn,8)
  l=nn+nn
  DO i=1,l,2
    angle=0.5_8*REAL(i-1,8)*del
    trigs(i)=COS(angle)
    trigs(i+1)=SIN(angle)
  END DO
  IF (imode == 1) RETURN
  IF (imode == 8) RETURN
  del=0.5_8*del
  nh=(nn+1)/2
  l=nh+nh
  la=nn+nn
  DO i=1,l,2
    angle=0.5_8*REAL(i-1,8)*del
    trigs(la+i)=COS(angle)
    trigs(la+i+1)=SIN(angle)
  END DO
  IF (imode <= 3) RETURN
  del=0.5_8*del
  la=la+nn
  IF (mode == 5) GO TO 40
  DO i=2,nn
    angle=REAL(i-1,8)*del
    trigs(la+i)=2.0_8*SIN(angle)
  END DO
  RETURN
  40 CONTINUE
  del=0.5_8*del
  DO i=2,n
    angle=REAL(i-1,8)*del
    trigs(la+i)=SIN(angle)
  END DO
  RETURN
END SUBROUTINE fftrig
SUBROUTINE vpassm(a,b,c,d,trigs,inc1,inc2,inc3,inc4,lot,n,ifac,la)
  implicit none
  integer, intent(in) :: inc1, inc2, inc3, inc4, lot, n, ifac, la
  real(8), intent(in) :: a(n), trigs(n)
  real(8), intent(out) :: b(n), c(n), d(n)

  integer :: i, ia, ib, ic, id, ie, iink, igo, ibase, ijk, &
          &  j, ja, jb, jc, jd, je, jink, jump, jbase, &
          &  k, kb, kc, kd, ke, l, la1, m
  real(8) :: c1, c2, c3, c4, s1, s2, s3, s4

  real(8), parameter :: sin36 = 0.587785252292473_8
  real(8), parameter :: cos36 = 0.809016994374947_8
  real(8), parameter :: sin72 = 0.951056516295154_8
  real(8), parameter :: cos72 = 0.309016994374947_8
  real(8), parameter :: sin60 = 0.866025403784437_8
  real(8), parameter :: half  = 0.5_8
!
!  SUBROUTINE "VPASSM" - MULTIPLE VERSION OF "VPASSA"
!  PERFORMS ONE PASS THROUGH DATA
!  AS PART OF MULTIPLE COMPLEX FFT ROUTINE
!  A IS FIRST REAL INPUT VECTOR
!  B IS FIRST IMAGINARY INPUT VECTOR
!  C IS FIRST REAL OUTPUT VECTOR
!  D IS FIRST IMAGINARY OUTPUT VECTOR
!  TRIGS IS PRECALCULATED TABLE OF SINES " COSINES
!  INC1 IS ADDRESSING INCREMENT FOR A AND B
!  INC2 IS ADDRESSING INCREMENT FOR C AND D
!  INC3 IS ADDRESSING INCREMENT BETWEEN A"S & B"S
!  INC4 IS ADDRESSING INCREMENT BETWEEN C"S & D"S
!  LOT IS THE NUMBER OF VECTORS
!  N IS LENGTH OF VECTORS
!  IFAC IS CURRENT FACTOR OF N
!  LA IS PRODUCT OF PREVIOUS FACTORS
!
!
  m=n/ifac
  iink=m*inc1
  jink=la*inc2
  jump=(ifac-1)*jink
  ibase=0
  jbase=0
  igo=ifac-1
  IF (igo > 4) RETURN
!  GO TO (10,50,90,130),igo
!
!  obsolescent feature correction temporarily by WYH.
!
   SELECT CASE (igo)
   CASE (1)
     GO TO 10
   CASE (2)
     GO TO 50
   CASE (3)
     GO TO 90
   CASE (4)
     GO TO 130
   CASE DEFAULT
!    Do nothing
   END SELECT
!
!  end of correction by WYH.
!
!
!  CODING FOR FACTOR 2
!
  10 ia=1
  ja=1
  ib=ia+iink
  jb=ja+jink
  DO l=1,la
    i=ibase
    j=jbase
!DIR$ IVDEP
    DO ijk=1,lot
      c(ja+j)=a(ia+i)+a(ib+i)
      d(ja+j)=b(ia+i)+b(ib+i)
      c(jb+j)=a(ia+i)-a(ib+i)
      d(jb+j)=b(ia+i)-b(ib+i)
      i=i+inc3
      j=j+inc4
    END DO
    ibase=ibase+inc1
    jbase=jbase+inc2
  END DO
  IF (la == m) RETURN
  la1=la+1
  jbase=jbase+jump
  DO k=la1,m,la
    kb=k+k-2
    c1=trigs(kb+1)
    s1=trigs(kb+2)
    DO l=1,la
      i=ibase
      j=jbase
!DIR$ IVDEP
      DO ijk=1,lot
        c(ja+j)=a(ia+i)+a(ib+i)
        d(ja+j)=b(ia+i)+b(ib+i)
        c(jb+j)=c1*(a(ia+i)-a(ib+i))-s1*(b(ia+i)-b(ib+i))
        d(jb+j)=s1*(a(ia+i)-a(ib+i))+c1*(b(ia+i)-b(ib+i))
        i=i+inc3
        j=j+inc4
      END DO
      ibase=ibase+inc1
      jbase=jbase+inc2
    END DO
    jbase=jbase+jump
  END DO
  RETURN
!
!  CODING FOR FACTOR 3
!
  50 ia=1
  ja=1
  ib=ia+iink
  jb=ja+jink
  ic=ib+iink
  jc=jb+jink
  DO l=1,la
    i=ibase
    j=jbase
!DIR$ IVDEP
    DO ijk=1,lot
      c(ja+j)=a(ia+i)+(a(ib+i)+a(ic+i))
      d(ja+j)=b(ia+i)+(b(ib+i)+b(ic+i))
      c(jb+j)=(a(ia+i)-half*(a(ib+i)+a(ic+i)))-(sin60*(b(ib+i)-b(ic+i)))
      c(jc+j)=(a(ia+i)-half*(a(ib+i)+a(ic+i)))+(sin60*(b(ib+i)-b(ic+i)))
      d(jb+j)=(b(ia+i)-half*(b(ib+i)+b(ic+i)))+(sin60*(a(ib+i)-a(ic+i)))
      d(jc+j)=(b(ia+i)-half*(b(ib+i)+b(ic+i)))-(sin60*(a(ib+i)-a(ic+i)))
      i=i+inc3
      j=j+inc4
    END DO
    ibase=ibase+inc1
    jbase=jbase+inc2
  END DO
  IF (la == m) RETURN
  la1=la+1
  jbase=jbase+jump
  DO k=la1,m,la
    kb=k+k-2
    kc=kb+kb
    c1=trigs(kb+1)
    s1=trigs(kb+2)
    c2=trigs(kc+1)
    s2=trigs(kc+2)
    DO l=1,la
      i=ibase
      j=jbase
!DIR$ IVDEP
      DO ijk=1,lot
        c(ja+j)=a(ia+i)+(a(ib+i)+a(ic+i))
        d(ja+j)=b(ia+i)+(b(ib+i)+b(ic+i))
        c(jb+j)=                                                        &
         &  c1*((a(ia+i)-half*(a(ib+i)+a(ic+i)))-(sin60*(b(ib+i)-b(ic+i)))) &
         &  -s1*((b(ia+i)-half*(b(ib+i)+b(ic+i)))+(sin60*(a(ib+i)-a(ic+i))))
        d(jb+j)=                                                        &
         &  s1*((a(ia+i)-half*(a(ib+i)+a(ic+i)))-(sin60*(b(ib+i)-b(ic+i)))) &
         &  +c1*((b(ia+i)-half*(b(ib+i)+b(ic+i)))+(sin60*(a(ib+i)-a(ic+i))))
        c(jc+j)=                                                        &
         &  c2*((a(ia+i)-half*(a(ib+i)+a(ic+i)))+(sin60*(b(ib+i)-b(ic+i)))) &
         &  -s2*((b(ia+i)-half*(b(ib+i)+b(ic+i)))-(sin60*(a(ib+i)-a(ic+i))))
        d(jc+j)=                                                        &
         &  s2*((a(ia+i)-half*(a(ib+i)+a(ic+i)))+(sin60*(b(ib+i)-b(ic+i)))) &
         &  +c2*((b(ia+i)-half*(b(ib+i)+b(ic+i)))-(sin60*(a(ib+i)-a(ic+i))))
        i=i+inc3
        j=j+inc4
      END DO
      ibase=ibase+inc1
      jbase=jbase+inc2
    END DO
    jbase=jbase+jump
  END DO
  RETURN
!
!  CODING FOR FACTOR 4
!
  90 ia=1
  ja=1
  ib=ia+iink
  jb=ja+jink
  ic=ib+iink
  jc=jb+jink
  id=ic+iink
  jd=jc+jink
  DO l=1,la
    i=ibase
    j=jbase
!DIR$ IVDEP
    DO ijk=1,lot
      c(ja+j)=(a(ia+i)+a(ic+i))+(a(ib+i)+a(id+i))
      c(jc+j)=(a(ia+i)+a(ic+i))-(a(ib+i)+a(id+i))
      d(ja+j)=(b(ia+i)+b(ic+i))+(b(ib+i)+b(id+i))
      d(jc+j)=(b(ia+i)+b(ic+i))-(b(ib+i)+b(id+i))
      c(jb+j)=(a(ia+i)-a(ic+i))-(b(ib+i)-b(id+i))
      c(jd+j)=(a(ia+i)-a(ic+i))+(b(ib+i)-b(id+i))
      d(jb+j)=(b(ia+i)-b(ic+i))+(a(ib+i)-a(id+i))
      d(jd+j)=(b(ia+i)-b(ic+i))-(a(ib+i)-a(id+i))
      i=i+inc3
      j=j+inc4
    END DO
    ibase=ibase+inc1
    jbase=jbase+inc2
  END DO
  IF (la == m) RETURN
  la1=la+1
  jbase=jbase+jump
  DO k=la1,m,la
    kb=k+k-2
    kc=kb+kb
    kd=kc+kb
    c1=trigs(kb+1)
    s1=trigs(kb+2)
    c2=trigs(kc+1)
    s2=trigs(kc+2)
    c3=trigs(kd+1)
    s3=trigs(kd+2)
    DO l=1,la
      i=ibase
      j=jbase
!DIR$ IVDEP
      DO ijk=1,lot
        c(ja+j)=(a(ia+i)+a(ic+i))+(a(ib+i)+a(id+i))
        d(ja+j)=(b(ia+i)+b(ic+i))+(b(ib+i)+b(id+i))
        c(jc+j)=                                                        &
         &  c2*((a(ia+i)+a(ic+i))-(a(ib+i)+a(id+i)))                    &
         &  -s2*((b(ia+i)+b(ic+i))-(b(ib+i)+b(id+i)))
        d(jc+j)=                                                        &
         &  s2*((a(ia+i)+a(ic+i))-(a(ib+i)+a(id+i)))                    &
         &  +c2*((b(ia+i)+b(ic+i))-(b(ib+i)+b(id+i)))
        c(jb+j)=                                                        &
         &  c1*((a(ia+i)-a(ic+i))-(b(ib+i)-b(id+i)))                    &
         &  -s1*((b(ia+i)-b(ic+i))+(a(ib+i)-a(id+i)))
        d(jb+j)=                                                        &
         &  s1*((a(ia+i)-a(ic+i))-(b(ib+i)-b(id+i)))                    &
         &  +c1*((b(ia+i)-b(ic+i))+(a(ib+i)-a(id+i)))
        c(jd+j)=                                                        &
         &  c3*((a(ia+i)-a(ic+i))+(b(ib+i)-b(id+i)))                    &
         &  -s3*((b(ia+i)-b(ic+i))-(a(ib+i)-a(id+i)))
        d(jd+j)=                                                        &
         &  s3*((a(ia+i)-a(ic+i))+(b(ib+i)-b(id+i)))                    &
         &  +c3*((b(ia+i)-b(ic+i))-(a(ib+i)-a(id+i)))
        i=i+inc3
        j=j+inc4
      END DO
      ibase=ibase+inc1
      jbase=jbase+inc2
    END DO
    jbase=jbase+jump
  END DO
  RETURN
!
!  CODING FOR FACTOR 5
!
  130 ia=1
  ja=1
  ib=ia+iink
  jb=ja+jink
  ic=ib+iink
  jc=jb+jink
  id=ic+iink
  jd=jc+jink
  ie=id+iink
  je=jd+jink
  DO l=1,la
    i=ibase
    j=jbase
!DIR$ IVDEP
    DO ijk=1,lot
      c(ja+j)=a(ia+i)+(a(ib+i)+a(ie+i))+(a(ic+i)+a(id+i))
      d(ja+j)=b(ia+i)+(b(ib+i)+b(ie+i))+(b(ic+i)+b(id+i))
      c(jb+j)=(a(ia+i)+cos72*(a(ib+i)+a(ie+i))-cos36*(a(ic+i)+a(id+i))) &
       &  -(sin72*(b(ib+i)-b(ie+i))+sin36*(b(ic+i)-b(id+i)))
      c(je+j)=(a(ia+i)+cos72*(a(ib+i)+a(ie+i))-cos36*(a(ic+i)+a(id+i))) &
       &  +(sin72*(b(ib+i)-b(ie+i))+sin36*(b(ic+i)-b(id+i)))
      d(jb+j)=(b(ia+i)+cos72*(b(ib+i)+b(ie+i))-cos36*(b(ic+i)+b(id+i))) &
       &  +(sin72*(a(ib+i)-a(ie+i))+sin36*(a(ic+i)-a(id+i)))
      d(je+j)=(b(ia+i)+cos72*(b(ib+i)+b(ie+i))-cos36*(b(ic+i)+b(id+i))) &
       &  -(sin72*(a(ib+i)-a(ie+i))+sin36*(a(ic+i)-a(id+i)))
      c(jc+j)=(a(ia+i)-cos36*(a(ib+i)+a(ie+i))+cos72*(a(ic+i)+a(id+i))) &
       &  -(sin36*(b(ib+i)-b(ie+i))-sin72*(b(ic+i)-b(id+i)))
      c(jd+j)=(a(ia+i)-cos36*(a(ib+i)+a(ie+i))+cos72*(a(ic+i)+a(id+i))) &
       &  +(sin36*(b(ib+i)-b(ie+i))-sin72*(b(ic+i)-b(id+i)))
      d(jc+j)=(b(ia+i)-cos36*(b(ib+i)+b(ie+i))+cos72*(b(ic+i)+b(id+i))) &
       &  +(sin36*(a(ib+i)-a(ie+i))-sin72*(a(ic+i)-a(id+i)))
      d(jd+j)=(b(ia+i)-cos36*(b(ib+i)+b(ie+i))+cos72*(b(ic+i)+b(id+i))) &
       &  -(sin36*(a(ib+i)-a(ie+i))-sin72*(a(ic+i)-a(id+i)))
      i=i+inc3
      j=j+inc4
    END DO
    ibase=ibase+inc1
    jbase=jbase+inc2
  END DO
  IF (la == m) RETURN
  la1=la+1
  jbase=jbase+jump
  DO k=la1,m,la
    kb=k+k-2
    kc=kb+kb
    kd=kc+kb
    ke=kd+kb
    c1=trigs(kb+1)
    s1=trigs(kb+2)
    c2=trigs(kc+1)
    s2=trigs(kc+2)
    c3=trigs(kd+1)
    s3=trigs(kd+2)
    c4=trigs(ke+1)
    s4=trigs(ke+2)
    DO l=1,la
      i=ibase
      j=jbase
!DIR$ IVDEP
      DO ijk=1,lot
        c(ja+j)=a(ia+i)+(a(ib+i)+a(ie+i))+(a(ic+i)+a(id+i))
        d(ja+j)=b(ia+i)+(b(ib+i)+b(ie+i))+(b(ic+i)+b(id+i))
        c(jb+j)=                                                        &
         &  c1*((a(ia+i)+cos72*(a(ib+i)+a(ie+i))-cos36*(a(ic+i)+a(id+i))) &
         &    -(sin72*(b(ib+i)-b(ie+i))+sin36*(b(ic+i)-b(id+i))))       &
         &  -s1*((b(ia+i)+cos72*(b(ib+i)+b(ie+i))-cos36*(b(ic+i)+b(id+i))) &
         &    +(sin72*(a(ib+i)-a(ie+i))+sin36*(a(ic+i)-a(id+i))))
        d(jb+j)=                                                        &
         &  s1*((a(ia+i)+cos72*(a(ib+i)+a(ie+i))-cos36*(a(ic+i)+a(id+i))) &
         &    -(sin72*(b(ib+i)-b(ie+i))+sin36*(b(ic+i)-b(id+i))))       &
         &  +c1*((b(ia+i)+cos72*(b(ib+i)+b(ie+i))-cos36*(b(ic+i)+b(id+i))) &
         &    +(sin72*(a(ib+i)-a(ie+i))+sin36*(a(ic+i)-a(id+i))))
        c(je+j)=                                                        &
         &  c4*((a(ia+i)+cos72*(a(ib+i)+a(ie+i))-cos36*(a(ic+i)+a(id+i))) &
         &    +(sin72*(b(ib+i)-b(ie+i))+sin36*(b(ic+i)-b(id+i))))       &
         &  -s4*((b(ia+i)+cos72*(b(ib+i)+b(ie+i))-cos36*(b(ic+i)+b(id+i))) &
         &    -(sin72*(a(ib+i)-a(ie+i))+sin36*(a(ic+i)-a(id+i))))
        d(je+j)=                                                        &
         &  s4*((a(ia+i)+cos72*(a(ib+i)+a(ie+i))-cos36*(a(ic+i)+a(id+i))) &
         &    +(sin72*(b(ib+i)-b(ie+i))+sin36*(b(ic+i)-b(id+i))))       &
         &  +c4*((b(ia+i)+cos72*(b(ib+i)+b(ie+i))-cos36*(b(ic+i)+b(id+i))) &
         &    -(sin72*(a(ib+i)-a(ie+i))+sin36*(a(ic+i)-a(id+i))))
        c(jc+j)=                                                        &
         &  c2*((a(ia+i)-cos36*(a(ib+i)+a(ie+i))+cos72*(a(ic+i)+a(id+i))) &
         &    -(sin36*(b(ib+i)-b(ie+i))-sin72*(b(ic+i)-b(id+i))))       &
         &  -s2*((b(ia+i)-cos36*(b(ib+i)+b(ie+i))+cos72*(b(ic+i)+b(id+i))) &
         &    +(sin36*(a(ib+i)-a(ie+i))-sin72*(a(ic+i)-a(id+i))))
        d(jc+j)=                                                        &
         &  s2*((a(ia+i)-cos36*(a(ib+i)+a(ie+i))+cos72*(a(ic+i)+a(id+i))) &
         &    -(sin36*(b(ib+i)-b(ie+i))-sin72*(b(ic+i)-b(id+i))))       &
         &  +c2*((b(ia+i)-cos36*(b(ib+i)+b(ie+i))+cos72*(b(ic+i)+b(id+i))) &
         &    +(sin36*(a(ib+i)-a(ie+i))-sin72*(a(ic+i)-a(id+i))))
        c(jd+j)=                                                        &
         &  c3*((a(ia+i)-cos36*(a(ib+i)+a(ie+i))+cos72*(a(ic+i)+a(id+i))) &
         &    +(sin36*(b(ib+i)-b(ie+i))-sin72*(b(ic+i)-b(id+i))))       &
         &  -s3*((b(ia+i)-cos36*(b(ib+i)+b(ie+i))+cos72*(b(ic+i)+b(id+i))) &
         &    -(sin36*(a(ib+i)-a(ie+i))-sin72*(a(ic+i)-a(id+i))))
        d(jd+j)=                                                        &
         &  s3*((a(ia+i)-cos36*(a(ib+i)+a(ie+i))+cos72*(a(ic+i)+a(id+i))) &
         &    +(sin36*(b(ib+i)-b(ie+i))-sin72*(b(ic+i)-b(id+i))))       &
         &  +c3*((b(ia+i)-cos36*(b(ib+i)+b(ie+i))+cos72*(b(ic+i)+b(id+i))) &
         &    -(sin36*(a(ib+i)-a(ie+i))-sin72*(a(ic+i)-a(id+i))))
        i=i+inc3
        j=j+inc4
      END DO
      ibase=ibase+inc1
      jbase=jbase+inc2
    END DO
    jbase=jbase+jump
  END DO
  RETURN
END SUBROUTINE vpassm
