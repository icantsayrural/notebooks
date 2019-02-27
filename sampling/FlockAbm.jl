module FlockAbm
    using LinearAlgebra

    mutable struct Bird
        location::Array{Float64, 1}
        velocity::Array{Float64, 1}
    end

    function move!(b::Bird,
                   neighbours::Array{Bird, 1},
                   max_speed=2.,
                   max_force=0.05,
                   separation_weight=1.3,
                   alignment_weight=8.,
                   cohesion_weight=2.,
                   neighbour_radius=7.,
                   desired_separation=4.,
                   boundary_min=0.,
                   boundary_max=550.)
        accer = flock(b,
                      neighbours,
                      separation_weight,
                      alignment_weight,
                      cohesion_weight,
                      neighbour_radius,
                      desired_separation,
                      max_force,
                      max_speed)

        b.velocity = limit(b.velocity + accer, max_speed)

        b.location += b.velocity
        b.location[1] = wrap(b.location[1], boundary_min, boundary_max)
        b.location[2] = wrap(b.location[2], boundary_min, boundary_max)

        add_noise!(b.location, max_force)
        add_noise!(b.velocity, max_force)
    end

    function flock(b::Bird,
                   neighbours::Array{Bird, 1},
                   separation_weight::Float64,
                   alignment_weight::Float64,
                   cohesion_weight::Float64,
                   neighbour_radius::Float64,
                   desired_separation::Float64,
                   max_force::Float64,
                   max_speed::Float64)
        separation = separate(b, neighbours, desired_separation) * separation_weight
        alignment = align(b, neighbours, neighbour_radius, max_force) * alignment_weight
        cohesion = cohere(b, neighbours, neighbour_radius, max_speed, max_force) * cohesion_weight
        return separation + alignment + cohesion
    end

    function separate(b::Bird, neighbours::Array{Bird, 1}, desired_separation::Float64)
        mean = [0, 0]
        cnt = 0
        for n in neighbours
            d = norm(b.location - n.location)
            if d > 0 && d < desired_separation
                mean += normalize(b.location - n.location) / d
                cnt += 1
            end
        end

        if cnt > 0
            mean /= cnt
        end
        return mean
    end

    function align(b::Bird, neighbours::Array{Bird, 1}, neighbour_radius::Float64, max_force::Float64)
        mean = [0., 0.]
        cnt = 0
        for n in neighbours
            d = norm(n.location - b.location)
            if d > 0 && d < neighbour_radius
                mean += n.velocity
                cnt += 1
            end
        end

        if cnt > 0
            mean /= cnt
        end
        mean = limit(mean, max_force)
        return mean
    end

    function cohere(b::Bird, neighbours::Array{Bird, 1}, neighbour_radius::Float64, max_speed::Float64, max_force::Float64)
        sum = [0., 0.]
        cnt = 0
        for n in neighbours
            d = norm(n.location - b.location)
            if d > 0 && d < neighbour_radius
                sum += n.location
                cnt += 1
            end
        end
        if cnt > 0
            return steer_to(b, sum / cnt, max_speed, max_force)
        else
            return sum
        end
    end

    function steer_to(b::Bird, target::Array{Float64, 1}, max_speed::Float64, max_force::Float64)
        desired = target - b.location
        d = norm(desired)

        if d > 0
            desired = normalize(desired)

            if d < 100.
                desired .*= max_speed * (d / 100.)
            else
                desired .*= max_speed
            end

            steer = limit(desired - b.velocity, max_force)
        else
            steer = [0, 0]
        end

        return steer
    end

    function limit(vec::Array{Float64, 1}, max_value::Float64)
        if norm(vec) > max_value
            return normalize(vec) * max_value
        else
            return vec
        end
    end

    function add_noise!(value::Array{Float64, 1}, max_force::Float64)
        value .+= max_force * randn()
    end

    function wrap(val::Float64, boundary_min::Float64, boundary_max::Float64)
        if val <= boundary_min
            return boundary_max
        end
        if val >= boundary_max
            return boundary_min
        end
        return val
    end

end
