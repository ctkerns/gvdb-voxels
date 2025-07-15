
//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2018 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

// GVDB library
#include "gvdb.h"
using namespace nvdb;

// Fluid System
#include "fluid_system.h"

// Sample utils
#include "main.h"   // window system
#include "nv_gui.h" // gui system
#include <GL/glew.h>
#include <cstring>

#include "string_helper.h"

VolumeGVDB gvdb;
FluidSystem fluid;

#ifdef USE_OPTIX
// OptiX scene
#include "optix_scene.h"
OptixScene optx;
#endif

class Sample : public NVPWindow {
public:
  Sample();
  virtual bool init();
  virtual void display();
  virtual void reshape(int w, int h);
  virtual void motion(int x, int y, int dx, int dy);
  virtual void keyboardchar(unsigned char key, int mods, int x, int y);
  virtual void mouse(NVPWindow::MouseButton button,
                     NVPWindow::ButtonAction state, int mods, int x, int y);
  virtual void on_arg(std::string arg, std::string val);

  void clear_gvdb();
  void render_update();
  void render_frame();
  void draw_points();
  void draw_topology(); // draw gvdb topology
  void simulate();
  void upload_points(); // upload points to gpu
  void start_guis(int w, int h);
  void ClearOptix();
  void RebuildOptixGraph(int shading);
  void ReportMemory();

  int m_radius;
  Vector3DF m_origin;
  float m_renderscale;

  int m_w, m_h;
  int m_numpnts;
  DataPtr m_pnts;
  int m_shade_style;
  int gl_screen_tex;
  int mouse_down;
  bool m_render_optix;
  bool m_show_points;
  bool m_show_topo;

  std::string m_infile;

  bool m_info;
};

Sample sample_obj;

void handle_gui(int gui, float val) {
  switch (gui) {
  case 3: { // Shading gui changed
    float alpha =
        (val == 4)
            ? 0.03f
            : 0.8f; // when in volume mode (#4), make volume very transparent
    gvdb.getScene()->LinearTransferFunc(0.00f, 0.50f, Vector4DF(0, 0, 1, 0),
                                        Vector4DF(0.0f, 1, 0, 0.1f));
    gvdb.getScene()->LinearTransferFunc(0.50f, 1.0f,
                                        Vector4DF(0.0f, 1, 0, 0.1f),
                                        Vector4DF(1.0f, .0f, 0, 0.1f));
    gvdb.CommitTransferFunc();
  } break;
  }
}

void Sample::start_guis(int w, int h) {
  clearGuis();
  setview2D(w, h);
  guiSetCallback(handle_gui);
  addGui(10, h - 30, 130, 20, "Points", GUI_CHECK, GUI_BOOL, &m_show_points, 0,
         1.0f);
  addGui(150, h - 30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo, 0,
         1.0f);
}

void Sample::ClearOptix() { optx.ClearGraph(); }

void Sample::RebuildOptixGraph(int shading) {
  char filepath[1024];

  optx.ClearGraph();

  int m_mat_surf1 =
      optx.AddMaterial("optix_trace_surface", "trace_surface", "trace_shadow");
  MaterialParams *matp = optx.getMaterialParams(m_mat_surf1);
  matp->light_width = 1.2f;
  matp->shadow_width = 0.1f;
  matp->shadow_bias = 0.5f;
  matp->amb_color = Vector3DF(.05f, .05f, .05f);
  matp->diff_color = Vector3DF(.7f, .7f, .7f);
  matp->spec_color = Vector3DF(1.f, 1.f, 1.f);
  matp->spec_power = 400.0;
  matp->env_color = Vector3DF(0.f, 0.f, 0.f);
  matp->refl_width = 0.5f;
  matp->refl_bias = 0.5f;
  matp->refl_color = Vector3DF(0.4f, 0.4f, 0.4f);

  matp->refr_width = 0.0f;
  matp->refr_color = Vector3DF(0.1f, .1f, .1f);
  matp->refr_ior = 1.1f;
  matp->refr_amount = 0.5f;
  matp->refr_offset = 50.0f;
  matp->refr_bias = 0.5f;
  optx.SetMaterialParams(m_mat_surf1, matp);

  if (gvdb.FindFile("sky.png", filepath))
    optx.CreateEnvmap(filepath);

  // Add GVDB volume to the OptiX scene
  nvprintf("Adding GVDB Volume to OptiX graph.\n");
  char isect;
  switch (shading) {
  case SHADE_TRILINEAR:
    isect = 'S';
    break;
  case SHADE_VOLUME:
    isect = 'D';
    break;
  case SHADE_LEVELSET:
    isect = 'L';
    break;
  case SHADE_EMPTYSKIP:
    isect = 'E';
    break;
  }
  // Get the dimensions of the volume by looking at how the fluid system was
  // initialized (since at this moment, gvdb.getVolMin and gvdb.getVolMax
  // are both 0).
  Vector3DF volmin = gvdb.getVolMin();
  Vector3DF volmax = gvdb.getVolMax();
  Matrix4F xform = gvdb.getTransform();
  int atlas_glid = gvdb.getAtlasGLID(0);
  optx.AddVolume(atlas_glid, volmin, volmax, xform, m_mat_surf1, isect);

  // Set Transfer Function (once before validate)
  Vector4DF *src = gvdb.getScene()->getTransferFunc();
  optx.SetTransferFunc(src);

  // Validate OptiX graph
  nvprintf("Validating OptiX.\n");
  optx.ValidateGraph();

  // Assign GVDB data to OptiX
  nvprintf("Update GVDB Volume.\n");
  optx.UpdateVolume(&gvdb);

  nvprintf("Rebuild Optix.. Done.\n");
}

Sample::Sample() {
  m_renderscale = 2.0;
  m_infile = "teapot.scn";
}

void Sample::on_arg(std::string arg, std::string val) {
  if (arg.compare("-in") == 0) {
    m_infile = val;
    nvprintf("input: %s\n", m_infile.c_str());
  }

  if (arg.compare("-info") == 0) {
    nvprintf("print gvdb info\n");
    m_info = true;
  }

  if (arg.compare("-scale") == 0) {
    m_renderscale = strToNum(val);
    nvprintf("render scale: %f\n", m_renderscale);
  }
}

bool Sample::init() {
  m_w = getWidth(); // window width & height
  m_h = getHeight();
  mouse_down = -1;
  gl_screen_tex = -1;
  m_show_topo = false;
  m_radius = 1;
  m_origin = Vector3DF(0, 0, 0);
  m_shade_style = 5;

  m_show_points = false;
  m_render_optix = true;

  init2D("arial");

  // Initialize Optix Scene
  if (m_render_optix) {
    optx.InitializeOptix(m_w, m_h);
  }

  gvdb.SetDebug(false);
  gvdb.SetVerbose(false);
  gvdb.SetProfile(false, true);
  gvdb.SetCudaDevice(m_render_optix ? GVDB_DEV_CURRENT : GVDB_DEV_FIRST);
  gvdb.Initialize();
  gvdb.StartRasterGL();

  // Default Camera
  Camera3D *cam = new Camera3D;
  cam->setFov(50.0);
  cam->setNearFar(1, 10000);
  cam->setOrbit(Vector3DF(7, 27, 0), Vector3DF(0, 0, 0), 200, 1.0);
  gvdb.getScene()->SetCamera(cam);

  // Default Light
  Light *lgt = new Light;
  lgt->setOrbit(Vector3DF(-186, 128, 0), Vector3DF(2250, 220, 2220), 4000, 1.0);
  gvdb.getScene()->SetLight(0, lgt);

  // Default volume params
  gvdb.getScene()->SetSteps(0.25f, 16, 0.25f);        // Set raycasting steps
  gvdb.getScene()->SetExtinct(-1.0f, 1.1f, 0.0f);     // Set volume extinction
  gvdb.getScene()->SetVolumeRange(0.0f, -1.0f, 3.0f); // Set volume value range
  gvdb.getScene()->SetCutoff(0.005f, 0.001f, 0.0f);
  gvdb.getScene()->SetBackgroundClr(0.1f, 0.2f, 0.4f, 1.0);

  // Add render buffer
  nvprintf("Output buffer: %d x %d\n", m_w, m_h);
  gvdb.AddRenderBuf(0, m_w, m_h, 4);

  // Resize window
  resize_window(m_w, m_h);

  // Create opengl texture for display
  glViewport(0, 0, m_w, m_h);
  createScreenQuadGL(&gl_screen_tex, m_w, m_h);

  // Configure
  gvdb.Configure(3, 3, 3, 3, 4);
  gvdb.SetChannelDefault(32, 32, 1);
  gvdb.AddChannel(0, T_FLOAT, 1, F_LINEAR);
  gvdb.FillChannel(0, Vector4DF(0, 0, 0, 0));

  // Initialize GUIs
  start_guis(m_w, m_h);

  clear_gvdb();

  // Initialize fluid sim.
  fluid.setup();
  m_numpnts = fluid.getPoints().size();
  gvdb.AllocData(m_pnts, m_numpnts, sizeof(Vector3DF), true);
  upload_points();

  render_update();

  // Rebuild the Optix scene graph with GVDB
  if (m_render_optix)
    RebuildOptixGraph(SHADE_LEVELSET);

  return true;
}

void Sample::reshape(int w, int h) {
  // Resize the opengl screen texture
  glViewport(0, 0, w, h);
  createScreenQuadGL(&gl_screen_tex, w, h);

  // Resize the GVDB render buffers
  gvdb.ResizeRenderBuf(0, w, h, 4);

  // Resize OptiX buffers
  if (m_render_optix)
    optx.ResizeOutput(w, h);

  // Resize 2D UI
  start_guis(w, h);

  postRedisplay();
}

void Sample::ReportMemory() {
  std::vector<std::string> outlist;
  gvdb.MemoryUsage("gvdb", outlist);
  for (int n = 0; n < outlist.size(); n++)
    nvprintf("%s", outlist[n].c_str());
}

void Sample::clear_gvdb() {
  // Clear
  DataPtr temp;
  gvdb.SetPoints(temp, temp, temp);
  gvdb.CleanAux();
}

void Sample::simulate() {
  // Run fluid simulation
  PERF_PUSH("Simulate");
  fluid.run();
  upload_points();
  render_update();

  PERF_POP();
}

void Sample::upload_points() {
  // Load input data
  auto fluid_pnts = fluid.getPoints();

  Vector3DF *pnts = (Vector3DF *)m_pnts.cpu;
  for (int i = 0; i < m_numpnts; i++) {
    pnts[i] = fluid_pnts[i];
  }
  gvdb.CommitData(m_pnts); // Commit to GPU

  DataPtr temp;
  gvdb.SetPoints(m_pnts, temp, temp);
}

void Sample::render_update() {
  // Rebuild GVDB Render topology
  PERF_PUSH("Dynamic Topology");
  gvdb.RebuildTopology(m_numpnts, m_radius * 2.0f, m_origin);
  gvdb.FinishTopology(false,
                      true); // false. no commit pool	false. no compute bounds
  gvdb.UpdateAtlas();
  PERF_POP();

  // Gather points to level set
  PERF_PUSH("Points-to-Voxels");
  gvdb.ClearChannel(0);

  int scPntLen = 0;
  int subcell_size = 4;
  gvdb.InsertPointsSubcell_FP16(subcell_size, m_numpnts,
                                static_cast<float>(m_radius), m_origin,
                                scPntLen);
  gvdb.GatherLevelSet_FP16(subcell_size, m_numpnts,
                           static_cast<float>(m_radius), m_origin, scPntLen, 0,
                           0);
  gvdb.UpdateApron(0, 3.0f);
  PERF_POP();

  if (m_render_optix) {
    PERF_PUSH("Update OptiX");
    optx.UpdateVolume(&gvdb); // GVDB topology has changed
    PERF_POP();
  }

  if (m_info) {
    ReportMemory();
    gvdb.Measure(true);
  }
}

void Sample::render_frame() {
  // Render frame
  gvdb.getScene()->SetCrossSection(m_origin, Vector3DF(0, 0, -1));

  int sh;
  switch (m_shade_style) {
  case 0:
    sh = SHADE_OFF;
    break;
  case 1:
    sh = SHADE_VOXEL;
    break;
  case 2:
    sh = SHADE_EMPTYSKIP;
    break;
  case 3:
    sh = SHADE_SECTION3D;
    break;
  case 4:
    sh = SHADE_VOLUME;
    break;
  case 5:
    sh = SHADE_LEVELSET;
    break;
  };

  if (m_render_optix) {
    // OptiX render
    PERF_PUSH("Raytrace");
    optx.Render(&gvdb, SHADE_LEVELSET, 0);
    PERF_POP();
    PERF_PUSH("ReadToGL");
    optx.ReadOutputTex(gl_screen_tex);
    PERF_POP();
  } else {
    // CUDA render
    PERF_PUSH("Raytrace");
    gvdb.Render(sh, 0, 0);
    PERF_POP();
    PERF_PUSH("ReadToGL");
    gvdb.ReadRenderTexGL(0, gl_screen_tex);
    PERF_POP();
  }
  renderScreenQuadGL(gl_screen_tex); // Render screen-space quad with texture
}

void Sample::draw_topology() {
  start3D(gvdb.getScene()->getCamera()); // start 3D drawing

  for (int lev = 0; lev < 5; lev++) { // draw all levels
    int node_cnt = static_cast<int>(gvdb.getNumNodes(lev));
    const Vector3DF &color = gvdb.getClrDim(lev);
    const Matrix4F &xform = gvdb.getTransform();

    for (int n = 0; n < node_cnt; n++) { // draw all nodes at this level
      Node *node = gvdb.getNodeAtLevel(n, lev);
      if (node->mFlags == 0)
        continue;

      Vector3DF bmin = gvdb.getWorldMin(node); // get node bounding box
      Vector3DF bmax = gvdb.getWorldMax(node); // draw node as a box
      drawBox3DXform(bmin, bmax, color, xform);
    }
  }

  end3D(); // end 3D drawing
}

void Sample::draw_points() {
  Vector3DF *fpos = (Vector3DF *)m_pnts.cpu;

  Vector3DF p1, p2;
  Vector3DF c;

  Camera3D *cam = gvdb.getScene()->getCamera();
  start3D(cam);
  for (int n = 0; n < m_numpnts; n++) {
    p1 = *fpos++;
    p2 = p1 + Vector3DF(0.01f, 0.01f, 0.01f);
    c = p1 / Vector3DF(256.0, 256, 256);
    drawLine3D(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, c.x, c.y, c.z, 1);
  }
  end3D();
}

void Sample::display() {
  clearScreenGL();

  simulate();

  // Render frame
  render_frame();

  glDisable(GL_DEPTH_TEST);
  glClearDepth(1.0);
  glClear(GL_DEPTH_BUFFER_BIT);

  if (m_show_points)
    draw_points();
  if (m_show_topo)
    draw_topology();

  draw3D();
  drawGui(0);
  draw2D();

  postRedisplay(); // Post redisplay since simulation is continuous
}

void Sample::motion(int x, int y, int dx, int dy) {
  // Get camera for GVDB Scene
  Camera3D *cam = gvdb.getScene()->getCamera();
  Light *lgt = gvdb.getScene()->getLight();
  bool shift = (getMods() & NVPWindow::KMOD_SHIFT); // Shift-key to modify light

  switch (mouse_down) {
  case NVPWindow::MOUSE_BUTTON_LEFT: {
    // Adjust orbit angles
    Vector3DF angs = (shift ? lgt->getAng() : cam->getAng());
    angs.x += dx * 0.2f;
    angs.y -= dy * 0.2f;
    if (shift)
      lgt->setOrbit(angs, lgt->getToPos(), lgt->getOrbitDist(),
                    lgt->getDolly());
    else
      cam->setOrbit(angs, cam->getToPos(), cam->getOrbitDist(),
                    cam->getDolly());
  } break;

  case NVPWindow::MOUSE_BUTTON_MIDDLE: {
    // Adjust target pos
    cam->moveRelative(float(dx) * cam->getOrbitDist() / 1000,
                      float(-dy) * cam->getOrbitDist() / 1000, 0);
  } break;

  case NVPWindow::MOUSE_BUTTON_RIGHT: {
    // Adjust dist
    float dist = (shift ? lgt->getOrbitDist() : cam->getOrbitDist());
    dist -= dy;
    if (shift)
      lgt->setOrbit(lgt->getAng(), lgt->getToPos(), dist, cam->getDolly());
    else
      cam->setOrbit(cam->getAng(), cam->getToPos(), dist, cam->getDolly());
  } break;
  }
}

void Sample::keyboardchar(unsigned char key, int mods, int x, int y) {
  switch (key) {
  case '1':
    m_show_points = !m_show_points;
    break;
  case '2':
    m_show_topo = !m_show_topo;
    break;
  };
}

void Sample::mouse(NVPWindow::MouseButton button, NVPWindow::ButtonAction state,
                   int mods, int x, int y) {
  if (guiHandler(button, state, x, y))
    return;
  mouse_down = (state == NVPWindow::BUTTON_PRESS) ? button : -1;
}

int sample_main(int argc, const char **argv) {
  return sample_obj.run("GVDB Sparse Volumes - gPointCloud Sample",
                        "pointcloud", argc, argv, 1280, 760, 4, 5, 30);
}

void sample_print(int argc, char const *argv) {}
