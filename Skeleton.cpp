//=============================================================================================
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Koncz Adam
// Neptun : MOENI1
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec4 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = color;	// computed color is the color of the primitive
	}
)";

const char * vertexSourceForBackground = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;														// copy texture coordinates
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

const char * fragmentSourceForBackground = R"(
	
	#version 330
    precision highp float;
	
	uniform float tension = 0.85f;
	 vec2 leftside;
	 vec2 actual;
	 vec2 rightside;
	 vec2 twoToTheRight;
	    vec2 controlPoints[8]=vec2[8](
		vec2(-0.1, 0.6),
		vec2(0.2, 0.7),
		vec2(0.4,0.6),
		vec2(0.6,0.9),
		vec2(0.8,0.65),
		vec2(1,0.4),
		vec2(1.2,0.6),
		vec2(1.4, 0.6)
		
);	

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 outColor;		// output that goes to the raster memory as told by glBindFragDataLocation

float H0(float s) {
		return 2*s*s*s - 3*s*s + 1;
}
float H1(float s) {
	return -2 * s*s*s + 3 * s*s ;
}
float H2(float s) {
	return  s*s*s - 2 * s*s + s;
}
float H3(float s) {
	return s*s*s - s*s;
}
	bool isHill(vec2 Coord){
		float y = 0;
		for(int i = 1; i < 7; i++){
			if(controlPoints[i].x > Coord.x){
				leftside = controlPoints[i-2];
				actual = controlPoints[i-1];
				rightside = controlPoints[i];
				twoToTheRight = controlPoints[i+1];
				break;
			}
		}
		float deltaI = rightside.x - actual.x;
		float TiI = ((1 - tension) / 2)*(twoToTheRight.y - rightside.y) + ((1 - tension)/ 2)*(rightside.y - actual.y);
		float TiO = ((1 - tension)/ 2)*(rightside.y - actual.y) + ((1 - tension)/ 2)*(actual.y - leftside.y);
		y = H0((Coord.x - actual.x) / deltaI) * actual.y;
		y += H1((Coord.x - actual.x) / deltaI) * rightside.y;
		y += H2((Coord.x - actual.x) / deltaI) * TiO;
		y += H3((Coord.x - actual.x) / deltaI) * TiI ;
		if(y > Coord.y)
			return true;
		return false;
	}


	void main() {
		if (isHill(texCoord)) {
			outColor = vec4(112/255.0f, 88/255.0f, 63/255.0f, 1); 
		} else {
			outColor = vec4(159/255.0f, 207/255.0f, 230/255.0f, 0.98f);
		}
	}
)";

enum  Orientation {
	left = -1,
	right = 1
};

GPUProgram gpuProgram; // vertex and fragment shaders
GPUProgram backGroundMaker;
unsigned int vao;	   // virtual world on the GPU

class Camera2D {
	vec2 wCenter; // center in world coordinates
	vec2 wSize;   // width and height in world coordinates
public:
	Camera2D() : wCenter(0, 0), wSize(20, 20) { }

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;		// 2D camera

#include <iostream>
	 void printvector(const std::vector<vec4>& asd) {
		 for (auto temp : asd)
			 std::cout << temp.x << ',' << temp.y << std::endl;
	 }

	 void printvector(const std::vector<float>& asd) {
		 for (int i = 0; i < asd.size()-4; i+=5)
			 std::cout << asd[i] << ',' << asd[i+1] << std::endl;
		 std::cout << "vector ends here" << std::endl;
	 }


int compareVec4ByX(const void* v1, const void* v2){
	int x1 = ((vec4*)v1)->x;
	int x2 = ((vec4*)v2)->x;
	if (x2 < x1)
		return 1;
	if (x2 == x1)
		return 0;
	return -1;
}

class KochanekBartelsSpline {
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;
	std::vector<float> ts;  // knots
	float tension, bias, continuity;
	float L(int i, float t) {
		float Li = 1.0f;
		for (unsigned int j = 0; j < wCtrlPoints.size(); j++)
			if (j != i) Li *= (t - ts[j]) / (ts[i] - ts[j]);
		return Li;
	}
	float H0(float);
	float H1(float);
	float H2(float);
	float H3(float);
protected:
	std::vector<vec4> wCtrlPoints;		// coordinates of control points
public:
	KochanekBartelsSpline(float t = 0.0f, float c = 0.0f, float b = 0.0f);

	void addSomeControlPoints(float y) {
		AddControlPointByCord(-19.0f, y);
		AddControlPointByCord(-17.0f, y);
		AddControlPointByCord(-15.0f, y);
		AddControlPointByCord(15.0f, y);
		AddControlPointByCord(17.0f, y);
		AddControlPointByCord(19.0f, y);
	 }

	 void AddControlPoint(float cX, float cY) {
		ts.push_back((float)wCtrlPoints.size());
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		wCtrlPoints.push_back(wVertex);
		//qsort(&wCtrlPoints[0], wCtrlPoints.size(), sizeof(vec4), compareVec4ByX);
		sortControlpoints();
	}

	 void AddControlPointByCord(float cX, float cY) {
		 ts.push_back((float)wCtrlPoints.size());
		 vec4 wVertex = vec4(cX, cY, 0, 1);
		 wCtrlPoints.push_back(wVertex);
		 sortControlpoints();
	 }


	void Draw() {
		mat4 VPTransform = camera.V() * camera.P();
		gpuProgram.Use();
		VPTransform.SetUniform(gpuProgram.getId(), "MVP");

		int colorLocation = glGetUniformLocation(gpuProgram.getId(), "color");

		if (wCtrlPoints.size() > 0) {	// draw control points
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, wCtrlPoints.size() * 4 * sizeof(float), &wCtrlPoints[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0) glUniform3f(colorLocation, 1, 0, 0);
				glPointSize(10.0f);
			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform4f(location, 1.0f, 0.0f, 0.0f, 0.0f); // 3 floats*/
			glDrawArrays(GL_POINTS, 0, wCtrlPoints.size());
		}
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform4f(location, 33/256.0f, 161/256.0f, 30/256.0f, 1.0f); // 3 floats
		if (wCtrlPoints.size() >= 2) {	// draw curve
			std::vector<float> vertexData;


			for (int x = 0; x < 400; x++) {
				float i = ((float)x / 20.0f) - 10.0f;
				vec4 wVertex(i, calculateY(i), 0, 1);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(-10.0f);
			}
			// copy data to the GPU
			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			if (colorLocation >= 0)
				glUniform3f(colorLocation, 1, 1, 0);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 800);

		}

	}
		void sortControlpoints() {
			for (int i = wCtrlPoints.size() - 1; i > 0; i--) {
				for (int j = 0; j < i - 1; j++) {
					if (wCtrlPoints[j].x > wCtrlPoints[j + 1].x) {
						vec4 Temp = wCtrlPoints[j];
						wCtrlPoints[j] = wCtrlPoints[j+1];
						wCtrlPoints[j + 1] = Temp;
					}
				}
			}
	}
		float calculateY(float x) {
			vec4* leftside = &wCtrlPoints[0];
			vec4* actual = &wCtrlPoints[1];
			vec4* rightside = nullptr;
			vec4* toTheToRight = nullptr;
			for (unsigned int i = 2; i < wCtrlPoints.size() - 2; i++) {
				if (wCtrlPoints[i].x > x ) {
					leftside = &wCtrlPoints[i-2];
					actual = &wCtrlPoints[i-1];
					rightside = &wCtrlPoints[i];
					toTheToRight = &wCtrlPoints[i + 1];
					break;
				}
			}

			if (rightside == nullptr) {
				return -6.2f;
				printf("gebasz");
			}
			//Choosing Tangent Vectors
			for (int i = 0; i < wCtrlPoints.size(); i++) {
				if (wCtrlPoints[i].x == leftside->x &&wCtrlPoints[i + 3].x != toTheToRight->x) {
					printf("leftside: %d", i);
				if (wCtrlPoints[i + 3].x != toTheToRight->x)
					printf("totheright: %d, to the right is not  correct! \n", i+3);

				}
			}
			for (int i = 0; i < wCtrlPoints.size()-1; i++) {
				if (wCtrlPoints[i].x > wCtrlPoints[i + 1].x) {
					printf("gebasz detected!\n");
					qsort(&wCtrlPoints[0], wCtrlPoints.size(), sizeof(vec4), compareVec4ByX);
					sortControlpoints();
					if (wCtrlPoints[i].x > wCtrlPoints[i + 1].x)
						printf("gebasz not fixed!!");
				}
			}

			float deltaI = rightside->x - actual->x;
			//incoming target vector
			//float TiI = ((1 - tension)*(1 + continuity)*(1 - bias) / 2)*(rightside->y - actual->y) + ((1 - tension)*(1 - continuity)*(1 + bias) / 2)*(actual->y - leftside->y);
			float TiI = ((1 - tension)*(1 + continuity)*(1 - bias) / 2)*(toTheToRight->y - rightside->y) + ((1 - tension)*(1 - continuity)*(1 + bias) / 2)*(rightside->y - actual->y);

			//outgoing target vec
			float TiO = ((1 - tension)*(1 - continuity)*(1 - bias) / 2)*(rightside->y - actual->y) + ((1 - tension)*(1 + continuity)*(1 + bias) / 2)*(actual->y - leftside->y);

			float res = 0;
			res += H0((x - actual->x) / deltaI)  *  actual->y;
			res += H1((x - actual->x) / deltaI)  *  rightside->y;
			res += H2((x - actual->x) / deltaI)  * 1 * TiO;
			res += H3((x - actual->x) / deltaI)  * 1 * TiI ;
			float asdasdasd;
			return res;
			
		}
		float CalculateDeriative(float x) {
			float y = calculateY(x);
			float yp1 = calculateY(x + 0.01f);
			return -(yp1 - y) / (x - (x + 0.01f));
		}

};
KochanekBartelsSpline::KochanekBartelsSpline(float t, float c, float b):
	tension(t), continuity(c), bias(b){
	glGenVertexArrays(1, &vaoCurve);
	glBindVertexArray(vaoCurve);

	glGenBuffers(1, &vboCurve); // Generate 1 vertex buffer object
	glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
	// Enable the vertex attribute arrays
	glEnableVertexAttribArray(0);  // attribute array 0
	// Map attribute array 0 to the vertex data of the interleaved vbo
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

	// Control Points
	glGenVertexArrays(1, &vaoCtrlPoints);
	glBindVertexArray(vaoCtrlPoints);

	glGenBuffers(1, &vboCtrlPoints); // Generate 1 vertex buffer object
	glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
	// Enable the vertex attribute arrays
	glEnableVertexAttribArray(0);  // attribute array 0
	// Map attribute array 0 to the vertex data of the interleaved vbo
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL); // attribute array, components/attribute, component type, normalize?, stride, offset

	addSomeControlPoints(-3.2f);
}

float KochanekBartelsSpline::H0(float s) {
		return 2*s*s*s - 3*s*s + 1;
		}
float KochanekBartelsSpline::H1(float s) {
	return -2 * s*s*s + 3 * s*s ;
}
float KochanekBartelsSpline::H2(float s) {
	return  s*s*s - 2 * s*s + s;
}
float KochanekBartelsSpline::H3(float s) {
	return s*s*s - s*s;
}

KochanekBartelsSpline* kb;




void addPointToBuffer(float x, float y, std::vector<float> &tarolo, vec3 rgb, GLuint vbo) {
		tarolo.push_back(x);
		tarolo.push_back(y);
		tarolo.push_back(rgb.x);
		tarolo.push_back(rgb.y);
		tarolo.push_back(rgb.z);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, tarolo.size() * sizeof(float), &tarolo[0], GL_DYNAMIC_DRAW);

	}
class TexturedQuad {
	unsigned int vao, vbo[2];
	vec2 vertices[4], uvs[4];
public:
	TexturedQuad() {
		vertices[0] = vec2(-10, -10); uvs[0] = vec2(0, 0);
		vertices[1] = vec2(10, -10);  uvs[1] = vec2(1, 0);
		vertices[2] = vec2(10, 10);   uvs[2] = vec2(1, 1);
		vertices[3] = vec2(-10, 10);  uvs[3] = vec2(0, 1);
	}
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		glGenBuffers(2, vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);	   // copy to that part of the memory which will be modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	
	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source

		mat4 MVPTransform = camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		MVPTransform.SetUniform(backGroundMaker.getId(), "MVP");


		

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

class Biker {
	GLuint				vao, kullovao, vbo, kullovbo, vaobody, vbobody;	// vertex array object, vertex buffer object
	std::vector<float>  vertexData, kullodata, bodydata; // interleaved data of coordinates and colors
	vec2			    wTranslate;
	vec2 center;
	float radius;
	vec4 offset = vec4(0, 0, 0, 0);
	float rotate = 0;
	Orientation orientation = right;
	KochanekBartelsSpline* kb;
	float CalculateRotation(vec2 asd) {
		float circumference = 2 * radius * M_PI;
		float motion = sqrtf(powf(asd.x, 2) + powf(asd.y, 2));
		return -2 * M_PI*motion / circumference;
	}
public:
	void setSpline(KochanekBartelsSpline* kb) {
		this->kb = kb;
	}
	void moveCenterByVec(vec2 asd) {
		orientation = asd.x > 0 ? right : left;
		center = center + asd;
		center.y = kb->calculateY(center.x)+0.69f;
		wTranslate = wTranslate + asd;
		rotate += orientation*CalculateRotation(asd);
	}
	void Create() {
		
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);


		{
			glGenVertexArrays(1, &kullovao);
			glBindVertexArray(kullovao);
			glGenBuffers(1, &kullovbo); // Generate 1 vertex buffer object
			glBindBuffer(GL_ARRAY_BUFFER, kullovbo);
			glEnableVertexAttribArray(0);  // attribute array 0
			glEnableVertexAttribArray(1);  // attribute array 1
			// Map attribute array 0 to the vertex data of the interleaved vbo
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
			// copy data to the GPU
			glBindBuffer(GL_ARRAY_BUFFER, kullovbo);
			glBufferData(GL_ARRAY_BUFFER, kullodata.size() * sizeof(float), &kullodata[0], GL_DYNAMIC_DRAW);
		}
		{
			glGenVertexArrays(1, &vaobody);
			glBindVertexArray(vaobody);
			glGenBuffers(1, &vbobody); // Generate 1 vertex buffer object
			glBindBuffer(GL_ARRAY_BUFFER, vbobody);
			// Enable the vertex attribute arrays
			glEnableVertexAttribArray(0);  // attribute array 0
			glEnableVertexAttribArray(1);  // attribute array 1
			// Map attribute array 0 to the vertex data of the interleaved vbo
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
			// Map attribute array 1 to the color data of the interleaved vbo
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

			// copy data to the GPU
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, bodydata.size() * sizeof(float), &bodydata[0], GL_DYNAMIC_DRAW);
		}


	
		center = vec2(-5.2, -2.45f);
		radius = 0.8f;
		makeCircle();
		makekullok();
		makeBody();
	}


	mat4 M() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wTranslate.x, wTranslate.y, 0, 1); // translation
	}
	mat4 Minv() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wTranslate.x, -wTranslate.y, 0, 1); // inverse translation
	}
	mat4 Mkullok() {
		return mat4(1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1); // translation
	}


	void AddPointByCord(float x, float y) {
		vertexData.push_back(x);
		vertexData.push_back(y);
		vertexData.push_back(0); // red
		vertexData.push_back(0); // green
		vertexData.push_back(0); // blue
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
		glutPostRedisplay();
	}

	void makeCircle() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBindVertexArray(vao);
		for (float i = 0.0f; i <= 2*M_PI; i += 2*M_PI / 100) {
			AddPointByCord(cos(i)*radius + center.x, sin(i)*radius + center.y);
		}
	}

	void makekullok() {
		glBindVertexArray(kullovao);
		for (float i = 0.0f; i <= 2 * M_PI; i += 2 * M_PI / 6) {
			kullodata.push_back(cos(i+rotate)*radius + center.x);
			kullodata.push_back(sin(i+rotate)*radius + center.y);
			kullodata.push_back(0); // red
			kullodata.push_back(0); // green
			kullodata.push_back(0); // blue
			// copy data to the GPU
			glBindBuffer(GL_ARRAY_BUFFER, kullovbo);
			glBufferData(GL_ARRAY_BUFFER, kullodata.size() * sizeof(float), &kullodata[0], GL_DYNAMIC_DRAW);
			kullodata.push_back(center.x);
			kullodata.push_back(center.y);
			kullodata.push_back(0); // red
			kullodata.push_back(0); // green
			kullodata.push_back(0); // blue
			// copy data to the GPU
			glBindBuffer(GL_ARRAY_BUFFER, kullovbo);
			glBufferData(GL_ARRAY_BUFFER, kullodata.size() * sizeof(float), &kullodata[0], GL_DYNAMIC_DRAW);
		}
	}
	void AddPoint(float cX, float cY) {
		// input pipeline
		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv() * Minv();
		// fill interleaved data
		vertexData.push_back(wVertex.x);
		vertexData.push_back(wVertex.y);
		vertexData.push_back(0); // red
		vertexData.push_back(0); // green
		vertexData.push_back(0); // blue
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
	}
	void makeBody() {
		glBindVertexArray(vaobody);
		
		makeLeg(left);
		makeLeg(right);
		addPointToBuffer(center.x + 0.3f, center.y + 2.18f, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(center.x - 0.3f, center.y + 2.18f, bodydata, vec3(0, 0, 0), vbobody);
 	}

	void makeLeg(Orientation leg) {
		float offset = M_PI / 2 * leg;
		float thigh = 1.6f;
		float shin = 1.7f;
		const vec2 P1 = vec2(center.x, center.y + 2.2f);
		const vec2 P2 = vec2(cos(rotate+ offset)*radius*0.75 + center.x, sin(rotate + offset)*radius*0.75 + center.y);
		float R = sqrtf(powf(P2.x - P1.x, 2) + powf(P2.y - P1.y, 2));
		float centerDx = P1.x - P2.x;
		float centerDy = P1.y - P2.y;
		float r = sqrtf(centerDx * centerDx + centerDy * centerDy);

		float r2d = r * r;
		float r4d = r2d * r2d;
		float rThighSquared = thigh * thigh;
		float rShinSquared = shin * shin;
		float a = (rThighSquared - rShinSquared) / (2 * r2d);
		float r2r2 = (rThighSquared - rShinSquared);
		float c = sqrtf(2 * (rThighSquared + rShinSquared) / r2d - (r2r2 * r2r2) / r4d - 1);

		float fx = (P1.x + P2.x) / 2 + a * (P2.x - P1.x);
		float gx = c * (P2.y - P1.y) / 2;

		float ix1 = fx + gx;
		float ix2 = fx - gx;

		float fy = (P1.y + P2.y) / 2 + a * (P2.y - P1.y);
		float gy = c * (P1.x - P2.x) / 2;

		float iy1 = fy + gy;
		float iy2 = fy - gy;

		vec2 asd = orientation == right ? vec2(ix2, iy2) : vec2(ix1, iy1);

		addPointToBuffer(P1.x, P1.y, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(asd.x, asd.y, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(P2.x, P2.y, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(asd.x, asd.y, bodydata, vec3(0, 0, 0), vbobody);
		addPointToBuffer(P1.x, P1.y, bodydata, vec3(0, 0, 0), vbobody);

	}



	void Draw() {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform4f(location, 0.0f, 0.0f, 0.0f, 1.0f);
		if (vertexData.size() > 0) {
			vertexData.clear();
			makeCircle();
			// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
			mat4 MVPTransform = Mkullok() * camera.V() * camera.P();
			MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
			
			glBindVertexArray(vao);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glLineWidth(3.0f);
			glDrawArrays(GL_LINE_LOOP, 0, vertexData.size() / 5);
			glLineWidth(2.0f);

		}

		if (kullodata.size() > 0) {
			// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
			kullodata.clear();
			makekullok();
			mat4 MVPTransform = Mkullok() * camera.V() * camera.P();
			MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
			glBindVertexArray(kullovao);
			glBindBuffer(GL_ARRAY_BUFFER,kullovbo);
			glBufferData(GL_ARRAY_BUFFER, kullodata.size() * sizeof(float), &kullodata[0], GL_DYNAMIC_DRAW);
			glLineWidth(3.0f);
			glDrawArrays(GL_LINE_STRIP, 0, kullodata.size() / 5);
			glLineWidth(2.0f);
		}

		if (bodydata.size() > 0) {
			// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
			bodydata.clear();
			makeBody();
			mat4 MVPTransform = Mkullok() * camera.V() * camera.P();
			MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

			glBindVertexArray(vaobody);
			glBindBuffer(GL_ARRAY_BUFFER, vbobody);
			glLineWidth(3.0f);
			glDrawArrays(GL_LINE_STRIP, 0, bodydata.size() / 5);
			glLineWidth(2.0f);
		}
		//printf("%f\n", kb->CalculateDeriative(center.x) * orientation);
	}
};


TexturedQuad quad;
Biker biker;

void onInitialization() {
	quad.Create();
	kb = new KochanekBartelsSpline(-1.0f, 0, 0);
	biker.Create();
	biker.setSpline(kb);
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f); // Width of lines in pixels

// Create objects by setting up their vertex data on the GPU

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
	
	//glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	// create program for the GPU  fragmentSourceForBackground
	gpuProgram.Create(vertexSource, fragmentSource, "outColor");
	backGroundMaker.Create(vertexSourceForBackground, fragmentSourceForBackground, "outColor");
	glutPostRedisplay();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(159/256.0f, 207/256.0f, 230/256.0f, 0.98f);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform4f(location, 1.0f, 1.0f, 0.0f,1.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
		                      0, 1, 0, 0,    // row-major!
		                      0, 0, 1, 0,
		                      0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	location = glGetUniformLocation(backGroundMaker.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location
	
	glBindVertexArray(vao);  // Draw call
	//glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 /*# Elements*/);
	backGroundMaker.Use();
	quad.Draw();
	gpuProgram.Use();
	kb->Draw();
	biker.Draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') {
		biker.moveCenterByVec(vec2(0.1,0.00f));
		glutPostRedisplay();
	}
	if (key == 'a') {
		biker.moveCenterByVec(vec2(-0.1, 0.00f));
		glutPostRedisplay();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	
	
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		kb->AddControlPoint(cX, cY);
		glutPostRedisplay();
	}
}

const float dt = 0.01f;
#include <chrono>
#include <thread>
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	//long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	static float tend = 0;
	float tstart = tend;
	float lastFrameTime = tstart;
	//std::this_thread::sleep_for(std::chrono::milliseconds(600));
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	for (float t = tstart; t < tend; t += dt) {
	}
}
