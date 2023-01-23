function [X,Y] = sincdata(type,N,var,fname,row,varargin);
%SINCDATA Generates one-dimensional sinc test data.
%
%	Description
%
%	[X,Y] = SINCDATA(TYPE,N,VAR,FNAME,ROW) - noisy realisation of Y =
%	SIN(X)/X.
%
%	The parameters:
%
%
%	 TYPE  - the type of the additive noise. Recognised types are:
%	'gauss', 'laplace', and 'posexp'.
%
%	 N  - the number of samples.
%
%	 VAR  - the noise (variance).     FNAME  - the name of the file where
%	to write the samples. If the   string is empty (or the function has
%	only three parameters), then no file   is written.
%
%	 ROW  - if this option is set, then the input positions X   are not
%	random but evenly spaced.
%
%
%	The input X is in [-XRANGE,XRANGE] and Y=YRANGE*SINX(X/XRANGE).
%
%	The default values of the scaling parameters are XRANGE=20 and
%	YRANGE=5.  These defaults can be altered by having two additional
%	parameters (7 in total) when calling SINCDATA:
%
%	[x,y] = sincdata(type,n,var,fname,row,range,yrange)
%
%	See also
%	OGP, OGPTRAIN, C_REG_GAUSS, C_REG_EXP, C_REG_LAPL, DEMOGP_REG, SINC2DATA
%

%	Copyright (c) Lehel Csato (2001-2004)

% Defining default parameters
MagnifY = 5;
MagnifX = 20;

if length(varargin)>0;
  MagnifX = varargin{1};
  if length(varargin)>1; MagnifY = varargin{2}; end
end;

if nargin<4;
  fname= []; row = 0;
elseif nargin<5;
  row  = 0;
end;
if nargin<3;
  error('Not enough arguments in calling SINCDATA');
end;

  
if row
  X = -1:(2/(N-1)):1;
else
  X = 2*rand(1,N)-1;
end;
X = X';
Y = MagnifY * sinc(2*X);
X = MagnifX * X;

switch upper(type);
 case 'GAUSS'		      % additive Gaussian noise
  Y = Y + sqrt(var)*randn(N,1);
 case 'LAPLACE'		      % symmetric exponential noise
  if var
    Y = Y + laplace([N,1])*var;
  end;
 case 'POSEXP'		      % nonsymmetric noise
  if var;
    Y = Y + laplace([N,1],0)*var;
  end;
 otherwise
  error('Unknown noise type');
end;

if length(fname)
  eval(['save ' fname ' X Y;']);
end;
