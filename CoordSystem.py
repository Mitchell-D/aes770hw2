from typing import MutableSequence
import numpy as np
from typing import Self

# :TODO: intersect an integer grid

class IntAxis:
    """
    The IntAxis class provides handy indexing tricks for expressing a bounded
    or infinite set of integers as a 3-vector of integers or None like:

    (start, stop, step)

    These three integers or None are sufficient for describing a linear
    function over the integers that m

    An axis can be thought of as a vector A=Nx+b


    --( BOUNDED BY DEFAULT )--

    These axes have default values at the origin,
    and as such are de facto bounded

    (None|0, +, +)  ->  increasing (0, UB)
    (+, None|0, -)  ->  decreasing (UB, 0)
    (-, None|0, +)  ->  increasing (LB, 0)
    (None|0, -, -)  ->  decreasing (0, LB)

    --( UNBOUNDED BY DEFAULT )--

    These axes have None values that represent an unbounded axis direction.

    An axis with None values for both start and stop has no constraints.
    (None, None, +|-) -> No upper or lower bounds

    If the needed constraint is provided by adding this axis to a known
    size, the inter

    (+, None, +)   ->  increasing (LB, inf)   need UB > LB for stop
    (-, None, -)   ->  decreasing (UB, -inf)  need LB < UB for stop
    (None, +, -)   ->  decreasing (inf, LB)   need UB > LB for start
    (None, -, +)   ->  increasing (-inf, UB)  need LB < UB for start

    --( EXPLICITLY UNBOUNDED )--

    If there are no bounds whatsoever, this can be used as a placeholder.

    (None, None, +) -> no upper or lower bounds
    (None, None, -) -> no upper or lower bounds

    --( EXPLICITLY BOUNDED )--

    When both stop and start are defined, Axes are bounded ONLY if their
    start and stop values have the same sign, and the step direction aligns
    with the relative magnitudes of start and stop, so that start iterates
    towards stop.

    If the condition relating stop to start is not met, an error will be
    raised because it is nonsense to iterate away from the stop value.

    (+ , + , +)  AND  stop > start
    (- , - , +)  AND  stop > start
    (+ , + , -)  AND  stop < start
    (- , - , -)  AND  stop < start

    --( RELATIVELY BOUNDED )--

    Axes with relative limits have integer start and stop values that have
    have opposite signs, which means they index relative to different
    boundaries of an undefined array

    This is because in such a case, they are indexed relative to seperate
    boundaries of an undefined array. This is useful for specifying the
    "edges" of an axis,

    (+ , - , +)  Valid if applied to (size > start-stop)  (   |-->| )
    (- , + , +)  Valid if applied to (size < stop-start)  ( |-->|   )
    (+ , - , -)  Valid if applied to (size < start-stop)  ( |<--|   )
    (- , + , -)  Valid if applied to (size > stop-start)  (  |<--|  )

    For example, if you apply the IntAxis(start=-8, stop=7, step=3)
    to an array like list(range(10)), the sequence would be determined like

    + ------- + --------------------------------------------- +
    |         |                                               |
    |  INPUT: |     0   1   2   3   4    5   6   7   8   9    |
    + ------- +            <----------------------------------|
    |         |           /                                   |
    |  START: |           |-8  -7  -6   -5  -4  -3  -2  -1    |
    + ------- +           |                                   |
    |         |           |---------------------->            |
    |   STOP: |     0   1 | 2   3   4    5   6   7 \ 8   9    |
    + ------- +           |                        |          |
    |         |           |->          |->         |          |
    |   STEP: |           | 2 \        | 5 \                  |
    + ------- +           |   |        |   |                  |
    |         |           |   |        |   |                  |
    |         |          (  2      ,     5  )                 |
    |         |               \        /                      |
    | OUTPUT: |                  (2, 5)                       |
    + ------- + --------------------------------------------- +
    """
    @staticmethod
    def _parse_constraint(constraint):
        """
        A coordinate constraint is an argument specified on a per-coordinate
        axis basis that specifies an n-dimensional index range.

        The returned 3-tuple is identical to the arguments for a python slice
        like (start, stop, skip) for integer or None inclusive starting
        indeces, integer or None exclusive ending indeces, and integer skip
        step size. This tuple is guaranteed to be valid for initializing an
        IntAxis object explicitly.

        defaults:
         - start defaults to 0 or +/- inf depending on stop/step
         - stop defaults to 0 or +/- inf depending on start/step
         - step defaults to +1

        bounded if:
        - step>0 and stop>start and stop*start>0
        - step<0 and stop<start and stop*start>0

        in summary the bounded conditions are:
         1. stop and start magnitudes match step direction
         2. stop and start are both on the same side of 0
        """
        ## Interpret no constraint as the full unbounded integer axis
        if constraint is None:
            return (0, None, 1)
        ## Interpret integers as single-value arguments
        if isinstance(constraint, (int, np.integer)):
            if constraint >= 0:
                return (constraint, constraint+1, 1)
            else:
                return (constraint, constraint-1, -1)
        ## Parse a tuple argument
        if type(constraint) in (list, tuple):
            ## must be (start, stop) or (start, stop, step); any can be None.
            constraint = list(constraint)
            assert len(constraint) in (2, 3)
            assert all([isinstance(c, (int, np.integer))
                        for c in constraint if not c is None])
            ## Default step size 1
            if len(constraint)==2:
                constraint = [*constraint, 1]

            ## Any IntAxis with step==None is the empty axis. Just return it.
            if constraint[2] is None:
                return constraint

            ## Parse the default None values
            if constraint[0] is None:
                if constraint[0]==constraint[1]:
                    if constraint[2] > 0:
                        return (0, None, constraint[2])
                    return (-1, None, constraint[2])
                elif constraint[2]>0 and constraint[1]>0:
                    constraint[0] = 0
                elif constraint[2]<0 and constraint[1]<0:
                    constraint[0] = -1
                else :
                    raise ValueError(constraint,
                                     "cannot support unbounded start "
                                     "iterating away from stop")
            ## If stop is None, set bounded defaults if iterating towards 0
            if constraint[1] is None:
                if constraint[2]>0 and constraint[0]<0:
                    constraint[1] = 0
                elif constraint[2]<0 and constraint[0]>0:
                    constraint[1] = 0
                #else :
                #    raise ValueError(constraint,
                #                     "cannot support unbounded start "
                #                     "iterating away from stop")
            return constraint
        raise TypeError(f"Couldn't parse constraint {constraint} "
                        "which must be one of (None, int, tuple). "
                        "See the internal docs for more info.")

    @staticmethod
    def _is_bounded(constraint):
        """
        A bounded IntAxis has no upper or lower bounds that are None,
        which indicates the lack of a boundary. A bounded IntAxis has
        a specific size, but may be offset.

        A bounded IntAxis has a size with no None values

        An equivalent condition for boundedness is:
        floor((stop-start)/step) > 0
        """
        start, stop, step = IntAxis._parse_constraint(constraint)

        ## Any bounded IntAxis with any default bounds parsed has no None.
        ## Also, any IntAxis with opposite-sign start and stop is conditional.
        if any(b is None for b in (start, stop, step)) or stop*start<0:
            return False
        ## The array can be bounded with a positive or negative step direction,
        ## as long as the stop is in the direction iterated towards from start.
        p_bounded = step>0 and stop>start
        n_bounded = step<0 and stop<start
        return p_bounded or n_bounded

    def __init__(self, constraint=None):
        self._c = IntAxis._parse_constraint(constraint)
    def __sub__(self, other:Self):
        if not self.bounded:
            raise ValueError(f"IntAxes {self} must be bounded to subtract")
        if not other.bounded:
            raise ValueError(f"IntAxes {other} must be bounded to subtract")

    @property
    def constraint(self):
        return self._c
    @property
    def components(self):
        """
        """
        add_id = IntAxis(0, 1, self.size)
        mult_id = IntAxis((0, self.size, 1))

        normal = IntAxis((0, self.size, 1))
        scale = IntAxis((0, self.size*self.step, self.step))
        offset = IntAxis((self.start, self.start+1, 1))

    @property
    def start(self):
        """ Inclusive index value where iteration starts """
        return self._c[0]
    @property
    def stop(self):
        """ Exclusive upper boundary index value where iteration stops """
        return self._c[1]
    @property
    def step(self):
        """ indexing step size for this domain, ie normal vector length """
        return self._c[2]
    @property
    def bounded(self):
        """ An axis is bounded if its size is a specific integer """
        return IntAxis._is_bounded(self._c)
    @property
    def conditional(self):
        """
        An axis is conditionally bounded if its start and stop values
        have the opposite sign
        """
        if self.stop is None:
            return False
        return self.start*self.stop<0

    @property
    def increasing(self):
        """ An axis is increasing if the step size is greater than 0 """
        return None if self.step is None else self.step > 0
    @property
    def positive(self):
        """
        An axis is positive if it has a start value greater than or equal to
        zero (ie relative to the start of an array), and is explicitly
        bounded or unbounded (ie not conditional).
        """

    @property
    def size(self):
        """
        The size of an axis is the positive integer number of elements in a
        bounded axis, or None if the axis isn't bounded.

        As long as the axis is bounded, it may have negative step sizes and
        start/stop indeces, but it still has a well-defined size.
        """
        if not self.bounded:
            return None
        return (self.stop-self.start)//self.step
    @property
    def as_slice(self):
        return slice(*self._c)
    @property
    def as_tuple(self):
        return tuple(self._c)

    def __repr__(self):
        """ Return a string formatted like 'start stop step' """
        s = [[str(a),"NB"][a is None] for a in self.as_tuple]
        return "("+",".join(s)+")"

    @staticmethod
    def disjoint(A, B):
        """
        Returns True if the boundaries of both IntAxis objects are disjoint,
        regardless of the step.
        """
        ## Arrays must be explicitly bounded or explicitly unbounded or scalar
        if A.conditional or B.conditional:
            raise ValueError(
                    "Can't check for disjoint condition with conditionally"
                    f"bounded array {A} or {B}")

        ## Empty sets are always disjoint since they both have zero size (?)
        if A.step is None or B.step is None:
            return IntAxis((None, None, None))

        ## since conditional axes aren't allowed, any positive and negative
        ## axes will be disjoint from each other.
        if A.start*B.start<0:
            return IntAxis((None, None, None))

        ## Scalar values are only not disjoint if they are the same number
        ## regardless of their step value (as long as it's not None)
        if A.start==A.stop or B.start==B.stop:
            if not A.start==B.start:
                return IntAxis((None, None, None))
            return A

        ## True if one array is increasing towards the other's start
        ## IntAxis are garunteed to have start defined as an integer.
        c1 = (A.step>0 and B.start>=A.start) \
                or (B.step>0 and A.start>=B.start) \
                or (A.step<0 and B.start<=A.start) \
                or (B.step<0 and B.start<=A.start) \
        ## True if opposite iteration direction
        if A.step*B.step < 0:
            ## If either is unbounded, c1 is sufficient for intersection
            if A.stop is None or B.stop is None:
                if c1:
                    return IntAxis((None, None, None))
                return IntAxis((A.start, A.stop, A.step))
            ## True if bounded, iterating in opposite directions, and disjoint.
            return not ((A.step>0 and B.stop<A.stop and B.start>=A.start)
                        or (A.step<0 and B.stop>A.stop and B.start>=A.start))

        ## Iterating in same direction
        else:
            ## Impossible for 2 unbounded axes increasing in the same
            ## direction to be disjoint
            if A.stop is None and B.stop is None:
                return False
            ## Return False if 1 axis is unbounded but overlaps the other
            if (A.stop is None and B.stop>A.start) \
                    or (B.stop is None and A.stop>B.start):
                return False
            ## Return False if 1 axis is unbounded with no overlap
            elif A.stop is None or B.stop is None:
                return (None, None, None)

        ## True if
        c2 = (A.stop >= B.start and A.start > B.start) \
                or (B.stop >= A.start and B.start > A.start)
        c3 = (A.stop <= B.start and A.start < B.start) \
                or (B.stop <= A.start and B.start < A.start)

        ## If both increasing
        return [c3,c2][A.step>0]

class CoordAxis(IntAxis):
    """
    A CoordAxis object maps a bounded IntAxis to a monotonic 1d array of data
    coordinates, so that a query in data coordinates can be converted to a
    corresponding integer.
    """
    @staticmethod
    def _validate_coord_array(coords):
        """
        Takes a list or array of coordinates, checks conditions, and returns
        a 1d array if all conditions were valid.

        Coordinate arrays must be monotonic and 1-dimensional. They normally
        should have more than 1 element, but may still have size 1.
        """
        coords = np.asarray(coords)
        assert len(coords.shape)==1 # coords currently must be 1d
        assert coords.size >= 1
        ## Allow single points; don't check if they're ascending
        if coords.size == 1:
            return coords
        dcoords = np.diff(coords)
        is_ascending = dcoords[0] > 0
        assert np.all(np.diff(coords))
        return coords

    def __init__(self, coords):
        """
        :@param coords:list, ndarray, etc of data
            coordinate values (ie wavelength in um, latitude in degrees),
            which must by monotonic in every dimension. A 1d list will
            usually suffice. Sometimes it's useful to have a 2d meshgrid for
            projections. I really don't recommend going further than that.
        """
        self._coords = self._validate_coord_array(coords)
        self.min = np.amin(self._coords)
        self.max = np.amax(self._coords)
        super().__init__((0, self._coords.size, 1))

    @property
    def ascending(self):
        """
        A CoordAxis which is ascending has a monotonic coordinate axis that
        increases in the same direction it is indexed. As such, this method
        returns True if the coordinates are increasing, False if they are
        descending, and None if this CoordinateAxis is a single point that is
        neither ascending nor descending.
        """
        if self._coords.size == 1:
            return None
        return self._coords[1]>self._coords[0]

    @property
    def coords(self):
        return tuple(self._coords)

    @property
    def bounds(self):
        """ 2-tuple of coordinate bounds in data coordinates """
        return (self.min, self.max)

    def range(self, start:float, end:float, as_slice=False,
              inclusive_end:bool=False):
        """
        Search for a range of values in data coordinates using an inclusive
        start and exclusive end value, and return a bounded IntAxis or slice
        depicting the axis indeces corresponding to the range.
        """
        if start is None:
            start = self.min
        if end is None:
            end = self.max+1
        ## Look for inclusive or exclusive points above or below the start
        ## point depending on the direction.
        idx0 = self.index(start, inclusive=True,
                          above=end>start).start
        idxf = self.index(end, inclusive=inclusive_end,
                          above=end<start).start
        ## Assume iteration by 1.
        return IntAxis((idx0,idxf)) if not as_slice else slice(idx0,idxf)

    def around(self, value, bound_err=True):
        """
        Returns IntAxis objects describing the points
        """
        ## p1 is below or equal to value
        p1 = self.index(value,inclusive=True,above=False,bound_err=bound_err)
        ## p2 is above but not equal to value
        p2 = self.index(value,inclusive=False,above=True,bound_err=bound_err)

        ## If the indeces are the same, the value is probably out of bounds
        if p1.start==p2.start:
            if bound_err:
                raise ValueError(
                        f"No coord points surrounding {value};\n"
                        f"Closest is {p1.start} at {self._coords[p1.start]}")
            return p1
        return IntAxis((p1.start, p2.stop))


    def index(self, value, inclusive=True, above=None, bound_err=True):
        """
        Convert a data value to a coordinate axis index by searching for
        nearby values, given optional constraints.

        By default, this method returns the point that is closest to the
        requested value in either the ascending or descending direction, but by
        setting above to True or False (default None), you can constrain the
        search to only return the nearest point that is above or below.

        :@param value: value in data coordinates (ie pascals, kilometers)
        :@param inclusive: if False, won't return an index with a corresponding
            coordinate which is exactly equal to the requested value. This is
            helpful for ensuring uniqueness when iterating over data
            boundaries that might not align with coordinate values
        :@param above: If True, return the nearest coordinate index above
            the target value; if False return the nearest below.
        :@param bound_err: If True, an error is raised if a value is requested
            outside of the default range.
        """
        if value < self.min or value > self.max:
            if bound_err:
                raise ValueError(
                        f"Data value {value} is out of bounds {self.bounds}")
            return IntAxis([[0,1],[self.size-1,self.size]][value < self.min])
        distance = self._coords-value
        min_idx = np.argmin(np.abs(distance))
        ## Return inclusive exact points
        if inclusive and distance[min_idx] == 0.:
            return IntAxis(min_idx)
        ## Return direction-agnostic point
        if above is None:
            return IntAxis(min_idx)
        ## Closest point is larger than query
        if distance[min_idx]>0:
            ## Return closest point when the coords are above the value
            if above:
                return IntAxis(min_idx)
            ## Handle single point out of bounds
            if self.size==1:
                if bound_err:
                    raise ValueError(f"This point {self._coords} is strictly "
                                     f"above the provided value {value}")
                else:
                    return IntAxis(None)
            ## since min/max bounds were imposed, the surrounding index should
            ## be the closest index with a coordinate value below the provided
            ## depending on the direction of increase of the coordinates.
            return IntAxis(min_idx+[-1,1][not self.ascending])
        ## Closest point is less than query
        elif distance[min_idx]<0:
            ## Return closest point when coords are above values
            if not above:
                return IntAxis(min_idx)
            ## Handle single point out of bounds
            if self.size==1:
                if bound_err:
                    raise ValueError(f"This point {self._coords} is strictly "
                                     f"above the provided value {value}")
                else:
                    return IntAxis(None)
            ## since min/max bounds were imposed, the surrounding index should
            ## be the closest index with a coordinate value above the provided
            ## depending on the direction of increase of the coordinates.
            return IntAxis(min_idx+[-1,1][self.ascending])

class CoordSystem:
    """
    CoordSystem is abstraction/specification for expressing discrete ranges of
    labeled data coordinates as a mapping to a bounded IntAxis. It provides
    simple methods for selecting data indeces based on simple searches in
    data units, and can be serialized as a hierarchy of nested tuples
    containing integers, strings, and None values.
    """
    def __init__(self, coord_arrays:list):
        """
        Initialize the coordinate system with a list of coordinate arrays
        """
        self._axes = [CoordAxis(A) for A in coord_arrays]
        assert all([ax.bounded for ax in self._axes])

    @property
    def shape(self):
        """
        Tuple containing the size of each axis in this CoordSystem, in the
        order that the corresponding labels are stored. The returned sizes are
        None if the axis isn't bounded

        Note that if this CoordSystem is applied to an other CoordSystem with
        the same labels but a different permutation, the shape may change.
        """
        return tuple(ax.size for ax in self._axes)

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return self.shape

    def __str__(self):
        return str(self.shape)

    def domain(self, idx:int=None):
        if idx is None:
            return [self.domain(i) for i in range(len(self._axes))]
        return (self._axes[idx].min, self._axes[idx].max)

    def axis(self, idx:int):
        """
        Return the CoordAxis or IntAxis object associated with the index.
        """
        return self._axes[idx]

    def add_axis(self, axis:CoordAxis, position=None):
        """
        Add an IntAxis or CoordAxis to this CoordSystem at the provided index
        (defaulting if None to index equal to ndim; effectively append as last)

        Adding an axis extends the shape of this CoordAxis by one dimension
        with length equal to the size of the provided axis.

        :@param axis: initialized CoordAxis or IntAxis
        :@param position: Optional position in the CoordAxis shape to place the
            new axis. Defaults to last entry.
        :@return None:
        """
        position = position if position is None else self.ndim
        assert 0 <= position <= self.ndim
        self._axes.insert(position, axis)

def unit_test(constraints={}):
    default_constraints = {
            "shorthand":[
                None,
                1,
                (9, 100),
                (-9, -100),
                (4, None),
                (None, 4),
                ],
            "unbounded_default":[
                (None, None, 1),
                (None, None, 5),
                ],
            "bounded_default":[
                (None, 1, 5),
                (None, -2, -5),
                (1, None, -5),
                (-1, None, 5),
                ],
            "unbounded":[
                (4, None, 3),
                (-8, None, -3),
                (None, 1, -1),
                (None, -1, 1),
                ],
            "condition_bounded":[
                (5, -2, 8),   ##  + - +
                (8, -3, -1),  ##  + - -
                (-1, 0, -1),  ##  - + -
                ],
            "explicit_bounded":[
                (1, 5, 2),    ##  + + +
                (-5, -1, 1),  ##  - - +
                (5, 1, -1),   ##  + + -
                (-1, -5, -1), ##  - - -
                ],
            "scalar":[
                (0, 0, 5),  ## scalar value equal to zero
                (5, 5, 0),  ## scalar value equal to 5
                (5, 5, 19), ## scalar value equal to 5
                ],
            "null":[
                (5, 5, None),       ## Null set
                (None, None, None), ## Null set
                ]
            }
    default_constraints.update(constraints)
    for k,args in default_constraints.items():
        print(k)
        axes = [IntAxis(a) for a in args]
        labels = [f"d{i:02}" for i in range(len(args))]
        for ax in axes:
            properties = [str(ax), ax.bounded, ax.increasing, ax.bounds,
                          ax.as_tuple]
            print(" ".join(str(p) for p in properties))

if __name__=="__main__":
    #cs = CoordSystem(d1=2, d2=(None,5), d5=(-8, -3, None), d3=(-4,None))
    #constraints = [(40,50), None, None, 10, None, (65,135), (None,2)]
    unit_test()
    exit(0)
    constraints = [(-40,-5, -1), (0,10), (0,15), 10, (-50,-31, 2),
                   (65,135), (-502, -901, -5)]
    labels = [f"d{i:02}" for i in range(len(constraints))]

    to_shape = (4000, 13, 17, 242, 245, 73, 8000)

    cs = CoordSystem(**dict(zip(labels,constraints)))

    #inter = cs.intersect(to_shape, [f"d{i:02}" for i in range(len(to_shape))])
    #print(inter.shape)
