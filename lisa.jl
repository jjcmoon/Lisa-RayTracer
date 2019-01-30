using Images
using StaticArrays
using LinearAlgebra

### TYPES ###

const Vec = SVector{3,Float64}


# color  = [Red, Green, Blue]
# aldebo = [diffuse multiplier, spectral multiplier, reflective multiplier, refractive multiplier]
struct Material
	refractive_index::Float64
	colour::Vec
	albedo::Array{Float64,1}
	specular_exp::Int64
end

struct Ray
	origin::Vec
	dir::Vec
end

abstract type Entity end

struct Sphere <: Entity
	center::Vec
	radius::Float64
	material::Material
end

struct Plane <: Entity
	center::Vec
	material::Material
end

struct Light
	center::Vec
	intensity::Float64
end

struct Scene
	entities::Array{Entity,1}
	lights::Array{Light,1}
	bg_colour::Vec
end

struct RenderSettings
	width::Int32
	height::Int32
	fov::Float64
	camera::Vec
end


### LINEAR ALGEBRA UTILITY ###

function reflect(v::Vec, n::Vec)
	return v - n * 2. * dot(v,n)
end


function refract(I::Vec, N::Vec, refractive_index::Float64)
	# Snell's law
    cosi = - max(-1., min(1., dot(I,N)))
    etai = 1
    etat = refractive_index
    n = N
    if (cosi < 0) # if the ray is inside the object, swap the indices and invert the normal to get the correct result
        cosi = -cosi
        etai, etat = etat, etai
        n = -N
    end
    eta = etai / etat
    k = 1 - eta*eta*(1 - cosi*cosi)
    return k < 0 ? Vec3f(0,0,0) : I*eta + n*(eta * cosi - sqrt(k))
end


### RAY/ENTITY GEOMETRY ###

function intersectEntity(sphere::Sphere, ray::Ray)
	vco = sphere.center .- ray.origin
	k = dot(ray.dir, vco)

	if k<0
		# sphere behind ray
		return Inf
	end

	temp = dot(vco,vco) - sphere.radius^2
	if k^2 < temp
		# ray doesn't hit sphere
		return Inf
	end

	return k - sqrt(k^2 - temp)
end


function intersectEntity(plane::Plane, ray::Ray)
	k = dot(plane.center-ray.origin, plane.center) / dot(ray.dir,plane.center)
	if k<0
		# plane behind ray
		return Inf
	end
	return k
end

function normalOn(sphere::Sphere, point::Vec)
	return normalize(point - sphere.center)
end

function normalOn(plane::Plane, point::Vec)
	return normalize(-plane.center)
end


function sceneIntersect(scene::Scene, ray::Ray)
	dist = Inf
	closest_ent = nothing
	for ent in scene.entities
		newdist = intersectEntity(ent, ray)
		if newdist < dist
			dist = newdist
			closest_ent = ent
		end
	end
	return dist, closest_ent
end


### RENDERING ###

function castRay(scene::Scene, ray::Ray, recDepth::Int64)
	dist, ent = sceneIntersect(scene,ray)
	if ent == nothing || recDepth == 0
		return scene.bg_colour
	end

	hit = ray.origin + dist*ray.dir
	normal = normalOn(ent, hit)
	mat = ent.material

	diffuse_intensity = 0.
	specular_intensity = 0.
	for light in scene.lights
		light_dir = normalize(light.center - hit)
		
		# offset hit so shadow doesn't intersect on entitiy itself
		shadow_orig = if (dot(light_dir,normal) < 0) hit - normal*1e-3 else hit + normal*1e-3 end
		shadow_dir  = normalize(light.center - shadow_orig)
		shdist,shent = sceneIntersect(scene, Ray(shadow_orig, shadow_dir))
		if shent != nothing && shdist < norm(light.center-shadow_orig)
			continue
		end

		distance_damper = 3000 / norm(light.center-shadow_orig)^2

		# Diffuse lighting
		diffuse_coeff = dot(normal,light_dir)
		diffuse_intensity += max(0,diffuse_coeff) * light.intensity * distance_damper

		# Specular lighting
		specular_coeff = dot(-reflect(-light_dir, normal),ray.dir)^mat.specular_exp
		specular_intensity += max(0., specular_coeff)*light.intensity * distance_damper
	end

	# Reflective lighting
	reflect_light = [0.,0.,0.]
	if mat.albedo[3]>0
		reflect_dir = reflect(ray.dir, normal)
		reflect_orig = if (dot(reflect_dir,normal) < 0) hit - normal*1e-3 else hit + normal*1e-3 end
		reflect_coeff = castRay(scene, Ray(reflect_orig, reflect_dir), recDepth-1)
		reflect_light = reflect_coeff * mat.albedo[3]
	end

	# Refractive lighting
	refract_light = [0.,0.,0.]
	if mat.albedo[4]>0
		refract_dir = normalize(refract(ray.dir, normal, mat.refractive_index))
		refract_orig = if (dot(refract_dir,normal) < 0) hit - normal*1e-3 else hit + normal*1e-3 end
		refract_coeff = castRay(scene, Ray(refract_orig, refract_dir), recDepth-1)
		refract_light = refract_coeff * mat.albedo[4]
	end

	diffuse_light = mat.colour*diffuse_intensity*mat.albedo[1]
	specular_light = [1.,1.,1.]*specular_intensity*mat.albedo[2]

	return diffuse_light + specular_light + reflect_light + refract_light
end

function renderScene(scene::Scene, st::RenderSettings)

	img = Array{Float64,3}(undef,3,st.height,st.width)

	for i in 1:st.width
		for j in 1:st.height
			x =  (2*(i + 0.5)/st.width  - 1)*tan(st.fov/2)*st.width/st.height
			y = -(2*(j + 0.5)/st.height - 1)*tan(st.fov/2)
			ray = Ray(st.camera, normalize(Vec(x, y, -1.)))
			img[:,j,i] = castRay(scene,ray,3)
		end
	end
	img = map(clamp01nan, img)
	save("julia.png", colorview(RGB, img))
end


const red    = Material(1. , [.3,.1,.1], [.9,.1, 0.,0.], 10)
const ivory  = Material(1. , [.4,.4,.3], [.6,.3, .1,0.], 50)
const blue   = Material(1. , [.2,.2,.5], [.5,.3, 0.,0.], 8)
const mirror = Material(1. , [1.,1.,1.], [.0,10.,.8,0.], 1250)
const glass  = Material(1.5, [.6,.7,.8], [0.,.5, .1,.8], 125)



function main()
	width = 2000
	height = 1500
	fov = π/3

	camera = [0.,0.,0.]

	sp1 = Sphere([-3,0,-16], 3, ivory)
	sp2 = Sphere([-1,-1.5,-12], 2, mirror)
	sp4 = Sphere([1.5,-0.5,-18], 3, red)
	sp3 = Sphere([7,5,-18], 4, mirror)

	pl1 = Plane([0.,-5.,0.],blue)
	pl2 = Plane([0.,0.,-50.],red)

	l1 = Light([-20, 20,  20], 1.5)
	l2 = Light([ 30, 50, -25], 1.8)
	l3 = Light([ 30, 20, 30], 1.7)

	entities = [sp1,sp2,sp3,sp4,pl1]
	lights = [l1,l2,l3]


	scene = Scene(entities, lights, [0,0,0])
	renderScene(scene, RenderSettings(width,height,fov,camera))
end

main()