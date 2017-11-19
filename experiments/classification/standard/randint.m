##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

## -*- texinfo -*-
## @deftypefn {Function File} {@var{b} = } randint (@var{n})
## @deftypefnx {Function File} {@var{b} = } randint (@var{n},@var{m})
## @deftypefnx {Function File} {@var{b} = } randint (@var{n},@var{m},@var{range})
## @deftypefnx {Function File} {@var{b} = } randint (@var{n},@var{m},@var{range},@var{seed})
##
## Generate a matrix of random binary numbers. The size of the matrix is
## @var{n} rows by @var{m} columns. By default @var{m} is equal to @var{n}.
##
## The range in which the integers are generated will is determined by
## the variable @var{range}. If @var{range} is an integer, the value will
## lie in the range [0,@var{range}]. If @var{range} contains two elements
## the intgers will lie between these two elements. By default @var{range}
## is assumed to be [0:1].
##
## The variable @var{seed} allows the random number generator to be seeded
## with a fixed value. The initial seed will be restored when returning.
## @end deftypefn

## 2001 FEB 07
##   initial release

function b = randint (n, m, range, seed)

  switch (nargin)
    case 1,
      m = n;
      range = [0,1];
      seed = Inf;
    case 2,
      range = [0,1];
      seed = Inf;
    case 3,
      seed = Inf;      
    case 4,
    otherwise
      usage ("b = randint (n, [m, [range, [seed]]])");
  endswitch

  ## Check range
  if (length (range) == 1)
    range = [0, range-1];
  elseif ( prod (size (range)) != 2)
    error ("randint: range must be a 2 element vector");
  endif
  range = sort (range);
  
  ## Check seed;
  if (!isinf (seed))
    old_seed = rand ("seed");
    rand ("seed", seed);
  endif

  b = range (1) - 1 + ceil (rand (n, m) * (range (2) - range (1) + 1));
  
  ## Get back to the old
  if (!isinf (seed))
    rand ("seed", old_seed);
  endif

endfunction
