/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ar.core.examples.java.rawdepth;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.Image;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.widget.SeekBar;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.ar.core.Anchor;
import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.DepthPoint;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.InstantPlacementPoint;
import com.google.ar.core.Plane;
import com.google.ar.core.Point;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.Trackable;
import com.google.ar.core.TrackingState;
import com.google.ar.core.examples.java.Mesh;
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper;
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper;
import com.google.ar.core.examples.java.common.helpers.FullScreenHelper;
import com.google.ar.core.examples.java.common.helpers.InstantPlacementSettings;
import com.google.ar.core.examples.java.common.helpers.SnackbarHelper;
import com.google.ar.core.examples.java.common.helpers.TapHelper;
import com.google.ar.core.examples.java.common.helpers.TrackingStateHelper;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;
import com.google.ar.sceneform.AnchorNode;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * This is a simple example that shows how to use ARCore Raw Depth API. The application will display
 * a 3D point cloud and allow the user control the number of points based on depth confidence.
 */
public class RawDepthActivity extends AppCompatActivity implements GLSurfaceView.Renderer {
  private static final String TAG = RawDepthActivity.class.getSimpleName();

  // Rendering. The Renderers are created here, and initialized when the GL surface is created.
  private GLSurfaceView surfaceView;

  private boolean installRequested;
  private boolean depthReceived;
  private TapHelper tapHelper;
  private final InstantPlacementSettings instantPlacementSettings = new InstantPlacementSettings();
  private final List<WrappedAnchor> wrappedAnchors = new ArrayList<>();
  private Mesh virtualObjectMesh;

  // Assumed distance from the device camera to the surface on which user will try to place objects.
  // This value affects the apparent scale of objects while the tracking method of the
  // Instant Placement point is SCREENSPACE_WITH_APPROXIMATE_DISTANCE.
  // Values in the [0.2, 2.0] meter range are a good choice for most AR experiences. Use lower
  // values for AR experiences where users are expected to place objects on surfaces close to the
  // camera. Use larger values for experiences where the user will likely be standing and trying to
  // place an object on the ground or floor in front of them.
  private static final float APPROXIMATE_DISTANCE_METERS = 2.0f;

  // Temporary matrix allocated here to reduce number of allocations for each frame.
  private final float[] modelMatrix = new float[16];
  private final float[] viewMatrix = new float[16];
  private final float[] projectionMatrix = new float[16];
  private final float[] modelViewMatrix = new float[16]; // view x model
  private final float[] modelViewProjectionMatrix = new float[16]; // projection x view x model
  private final float[] sphericalHarmonicsCoefficients = new float[9 * 3];
  private final float[] viewInverseMatrix = new float[16];
  private final float[] worldLightDirection = {0.0f, 0.0f, 0.0f, 0.0f};
  private final float[] viewLightDirection = new float[4]; // view x world light direction

  private Session session;
  private final SnackbarHelper messageSnackbarHelper = new SnackbarHelper();
  private DisplayRotationHelper displayRotationHelper;
  private final TrackingStateHelper trackingStateHelper = new TrackingStateHelper(this);

  private final Renderer renderer = new Renderer();

  // This lock prevents accessing the frame images while Session is paused.
  private final Object frameInUseLock = new Object();

  /** The current raw depth image timestamp. */
  private long depthTimestamp = -1;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    surfaceView = findViewById(R.id.surfaceview);
    displayRotationHelper = new DisplayRotationHelper(/*context=*/ this);

    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) !=
            PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA},
              50); }

    // Set up touch listener.
    tapHelper = new TapHelper(/*context=*/ this);
    surfaceView.setOnTouchListener(tapHelper);
    instantPlacementSettings.onCreate(this);

    // Set up rendering.
    surfaceView.setPreserveEGLContextOnPause(true);
    surfaceView.setEGLContextClientVersion(2);
    surfaceView.setEGLConfigChooser(8, 8, 8, 0, 16, 0);
    surfaceView.setRenderer(this);
    surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
    surfaceView.setWillNotDraw(false);

    // Set up confidence threshold slider.
    SeekBar seekBar = findViewById(R.id.slider);
    seekBar.setProgress((int) (renderer.getPointAmount() * seekBar.getMax()));
    seekBar.setOnSeekBarChangeListener(seekBarChangeListener);

    installRequested = false;
    depthReceived = false;
  }

  private SeekBar.OnSeekBarChangeListener seekBarChangeListener =
      new SeekBar.OnSeekBarChangeListener() {
        @Override
        public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
          float progressNormalized = (float) progress / seekBar.getMax();
          renderer.setPointAmount(progressNormalized);
        }

        @Override
        public void onStartTrackingTouch(SeekBar seekBar) {}

        @Override
        public void onStopTrackingTouch(SeekBar seekBar) {}
      };

  @Override
  protected void onDestroy() {
    if (session != null) {
      // Explicitly close ARCore Session to release native resources.
      // Review the API reference for important considerations before calling close() in apps with
      // more complicated lifecycle requirements:
      // https://developers.google.com/ar/reference/java/arcore/reference/com/google/ar/core/Session#close()
      session.close();
      session = null;
    }

    super.onDestroy();
  }

  @Override
  protected void onResume() {
    super.onResume();

    if (session == null) {
      Exception exception = null;
      String message = null;
      try {
        switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
          case INSTALL_REQUESTED:
            installRequested = true;
            return;
          case INSTALLED:
            break;
        }

        // ARCore requires camera permissions to operate. If we did not yet obtain runtime
        // permission on Android M and above, now is a good time to ask the user for it.
        if (!CameraPermissionHelper.hasCameraPermission(this)) {
          CameraPermissionHelper.requestCameraPermission(this);
          return;
        }

        // Create the session.
        session = new Session(/* context= */ this);
      } catch (UnavailableArcoreNotInstalledException
          | UnavailableUserDeclinedInstallationException e) {
        message = "Please install ARCore";
        exception = e;
      } catch (UnavailableApkTooOldException e) {
        message = "Please update ARCore";
        exception = e;
      } catch (UnavailableSdkTooOldException e) {
        message = "Please update this app";
        exception = e;
      } catch (UnavailableDeviceNotCompatibleException e) {
        message = "This device does not support AR";
        exception = e;
      } catch (RuntimeException e) {
        message = "Failed to create AR session";
        exception = e;
      }

      if (!session.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)) {
        message = "This device does not support the ARCore Raw Depth API.";
        session = null;
      }

      if (message != null) {
        messageSnackbarHelper.showError(this, message);
        Log.e(TAG, "Exception creating session", exception);
        return;
      }
    }

    // Note that order matters - see the note in onPause(), the reverse applies here.
    try {
      // Wait until the frame is no longer being processed.
      synchronized (frameInUseLock) {
        // Enable raw depth estimation and auto focus mode while ARCore is running.
        Config config = session.getConfig();
        config.setDepthMode(Config.DepthMode.RAW_DEPTH_ONLY);

        if (instantPlacementSettings.isInstantPlacementEnabled()) {
          config.setInstantPlacementMode(Config.InstantPlacementMode.LOCAL_Y_UP);
        } else {
          config.setInstantPlacementMode(Config.InstantPlacementMode.DISABLED);
        }

        session.configure(config);
        session.resume();
      }
    } catch (CameraNotAvailableException e) {
      messageSnackbarHelper.showError(this, "Camera not available. Try restarting the app.");
      session = null;
      return;
    }

    surfaceView.onResume();
    displayRotationHelper.onResume();

    messageSnackbarHelper.showMessage(this, "No depth yet. Try moving the device.");
  }

  @Override
  public void onPause() {
    super.onPause();
    if (session != null) {
      // Note that the order matters - see note in onResume().
      // GLSurfaceView is paused before pausing the ARCore session, to prevent onDrawFrame() from
      // calling session.update() on a paused session.
      displayRotationHelper.onPause();
      surfaceView.onPause();
      session.pause();
    }
  }

  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
    super.onRequestPermissionsResult(requestCode, permissions, results);
    if (!CameraPermissionHelper.hasCameraPermission(this)) {
      Toast.makeText(this, "Camera permission is needed to run this application", Toast.LENGTH_LONG)
          .show();
      if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
        // Permission denied with checking "Do not ask again".
        CameraPermissionHelper.launchPermissionSettings(this);
      }
      finish();
    }
  }

  @Override
  public void onWindowFocusChanged(boolean hasFocus) {
    super.onWindowFocusChanged(hasFocus);
    FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus);
  }

  @Override
  public void onSurfaceCreated(GL10 gl, EGLConfig config) {
    GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // Prepare the rendering objects. This involves reading shaders, so may throw an IOException.
    try {
      renderer.createOnGlThread(/*context=*/ this);

      virtualObjectMesh = Mesh.createFromAsset(getAssets(), "models/pawn.obj");

    } catch (IOException e) {
      Log.e(TAG, "Failed to read an asset file", e);
    }
  }

  @Override
  public void onSurfaceChanged(GL10 gl, int width, int height) {
    displayRotationHelper.onSurfaceChanged(width, height);
    GLES20.glViewport(0, 0, width, height);
  }

  @Override
  public void onDrawFrame(GL10 gl) {
    // Clear screen to notify driver it should not load any pixels from previous frame.
    GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);

    if (session == null) {
      return;
    }

    // Synchronize prevents session.update() call while paused, see note in onPause().
    synchronized (frameInUseLock) {
      // Notify ARCore that the view size changed so that the perspective matrix can be adjusted.
      displayRotationHelper.updateSessionIfNeeded(session);

      try {
        session.setCameraTextureNames(new int[] {0});

        Frame frame = session.update();
        Camera camera = frame.getCamera();

        // Handle one tap per frame.
        handleTap(frame, camera);



        // Keep the screen unlocked while tracking, but allow it to lock when tracking stops.
        trackingStateHelper.updateKeepScreenOnFlag(camera.getTrackingState());

        if (camera.getTrackingState() != TrackingState.TRACKING) {
          // If motion tracking is not available but previous depth is available, notify the user
          // that the app will resume with tracking.
          if (depthReceived) {
            messageSnackbarHelper.showMessage(
                this, TrackingStateHelper.getTrackingFailureReasonString(camera));
          }

          // If not tracking, do not render the point cloud.
          return;
        }

        // Check if the frame contains new depth data or a 3D reprojection of the previous data. See
        // documentation of acquireRawDepthImage16Bits for more details.
        boolean containsNewDepthData;
        try (Image depthImage = frame.acquireRawDepthImage16Bits()) {
          containsNewDepthData = depthTimestamp == depthImage.getTimestamp();
          depthTimestamp = depthImage.getTimestamp();
        } catch (NotYetAvailableException e) {
          // This is normal at the beginning of session, where depth hasn't been estimated yet.
          containsNewDepthData = false;
        }

        if (containsNewDepthData) {
          // Get Raw Depth data of the current frame.
          final DepthData depth = DepthData.create(session, frame);

          // Skip rendering the current frame if an exception arises during depth data processing.
          // For example, before depth estimation finishes initializing.
          if (depth != null) {
            depthReceived = true;
            renderer.update(depth);
          }
        }

        float[] projectionMatrix = new float[16];
        camera.getProjectionMatrix(projectionMatrix, 0, 0.1f, 100.0f);
        float[] viewMatrix = new float[16];
        camera.getViewMatrix(viewMatrix, 0);

        // Visualize depth points.
        renderer.draw(viewMatrix, projectionMatrix);

//        Pose mCameraRelativePose=Pose.makeTranslation(0.0f, 0.0f, -0.5f);
//        Anchor myAnchor = session.createAnchor(mCameraRelativePose);
////        AnchorNode anchorNode = new AnchorNode(myAnchor);
////        anchorNode.setParent(frame.;

//        //Add an Anchor and a renderable in front of the camera
//        float[] pos = { 0, 0, -1 };
//        float[] rotation = { 0, 0, 0, 1 };
//        Anchor anchor =  session.createAnchor(new Pose(pos, rotation));


        // Visualize anchors created by touch.
//        renderer.clear(virtualSceneFramebuffer, 0f, 0f, 0f, 0f);
        for (WrappedAnchor wrappedAnchor : wrappedAnchors) {
          Anchor anchor = wrappedAnchor.getAnchor();
          Trackable trackable = wrappedAnchor.getTrackable();
          if (anchor.getTrackingState() != TrackingState.TRACKING) {
            continue;
          }

          // Get the current pose of an Anchor in world space. The Anchor pose is updated
          // during calls to session.update() as ARCore refines its estimate of the world.
          anchor.getPose().toMatrix(modelMatrix, 0);

          // Calculate model/view/projection matrices
          Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0);
          Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0);

//          // Update shader properties and draw
//          virtualObjectShader.setMat4("u_ModelView", modelViewMatrix);
//          virtualObjectShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix);
//
//          if (trackable instanceof InstantPlacementPoint
//                  && ((InstantPlacementPoint) trackable).getTrackingMethod()
//                  == InstantPlacementPoint.TrackingMethod.SCREENSPACE_WITH_APPROXIMATE_DISTANCE) {
//            virtualObjectShader.setTexture(
//                    "u_AlbedoTexture", virtualObjectAlbedoInstantPlacementTexture);
//          } else {
//            virtualObjectShader.setTexture("u_AlbedoTexture", virtualObjectAlbedoTexture);
//          }


//          render.draw(virtualObjectMesh, virtualObjectShader, virtualSceneFramebuffer);
          virtualObjectMesh.lowLevelDraw();
        }


        // Hide all user notifications when the frame has been rendered successfully.
        messageSnackbarHelper.hide(this);
      } catch (Throwable t) {
        // Avoid crashing the application due to unhandled exceptions.
        Log.e(TAG, "Exception on the OpenGL thread", t);
      }
    }
  }


  // Handle only one tap per frame, as taps are usually low frequency compared to frame rate.
  private void handleTap(Frame frame, Camera camera) {
    MotionEvent tap = tapHelper.poll();
    if (tap != null && camera.getTrackingState() == TrackingState.TRACKING) {
      List<HitResult> hitResultList;
      if (instantPlacementSettings.isInstantPlacementEnabled()) {
        hitResultList =
                frame.hitTestInstantPlacement(tap.getX(), tap.getY(), APPROXIMATE_DISTANCE_METERS);
      } else {
        hitResultList = frame.hitTest(tap);
      }
      for (HitResult hit : hitResultList) {
        // If any plane, Oriented Point, or Instant Placement Point was hit, create an anchor.
        Trackable trackable = hit.getTrackable();
        // If a plane was hit, check that it was hit inside the plane polygon.
        // DepthPoints are only returned if Config.DepthMode is set to AUTOMATIC.
        if ((trackable instanceof Plane
                && ((Plane) trackable).isPoseInPolygon(hit.getHitPose())
                /* && (PlaneRenderer.calculateDistanceToPlane(hit.getHitPose(), camera.getPose()) > 0) */ )
                || (trackable instanceof Point
                && ((Point) trackable).getOrientationMode()
                == Point.OrientationMode.ESTIMATED_SURFACE_NORMAL)
                || (trackable instanceof InstantPlacementPoint)
                || (trackable instanceof DepthPoint)) {
          // Cap the number of objects created. This avoids overloading both the
          // rendering system and ARCore.
          if (wrappedAnchors.size() >= 20) {
            wrappedAnchors.get(0).getAnchor().detach();
            wrappedAnchors.remove(0);
          }

          // Adding an Anchor tells ARCore that it should track this position in
          // space. This anchor is created on the Plane to place the 3D model
          // in the correct position relative both to the world and to the plane.
          wrappedAnchors.add(new WrappedAnchor(hit.createAnchor(), trackable));
          // For devices that support the Depth API, shows a dialog to suggest enabling
          // depth-based occlusion. This dialog needs to be spawned on the UI thread.
//          this.runOnUiThread(this::showOcclusionDialogIfNeeded);

          // Hits are sorted by depth. Consider only closest hit on a plane, Oriented Point, or
          // Instant Placement Point.
          break;
        }
      }
    }
  }



  /** Configures the session with feature settings. */
  private void configureSession() {
    Config config = session.getConfig();
    config.setLightEstimationMode(Config.LightEstimationMode.ENVIRONMENTAL_HDR);
    if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
      config.setDepthMode(Config.DepthMode.AUTOMATIC);
    } else {
      config.setDepthMode(Config.DepthMode.DISABLED);
    }
    if (instantPlacementSettings.isInstantPlacementEnabled()) {
      config.setInstantPlacementMode(Config.InstantPlacementMode.LOCAL_Y_UP);
    } else {
      config.setInstantPlacementMode(Config.InstantPlacementMode.DISABLED);
    }
    session.configure(config);
  }


}


/**
 * Associates an Anchor with the trackable it was attached to. This is used to be able to check
 * whether or not an Anchor originally was attached to an {@link InstantPlacementPoint}.
 */
class WrappedAnchor {
  private Anchor anchor;
  private Trackable trackable;

  public WrappedAnchor(Anchor anchor, Trackable trackable) {
    this.anchor = anchor;
    this.trackable = trackable;
  }

  public Anchor getAnchor() {
    return anchor;
  }

  public Trackable getTrackable() {
    return trackable;
  }
}
