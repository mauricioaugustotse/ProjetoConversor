$ErrorActionPreference = "Stop"

$source = @"
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace AudioSessionsInspect {
  public enum EDataFlow { eRender = 0, eCapture = 1, eAll = 2 }
  public enum ERole { eConsole = 0, eMultimedia = 1, eCommunications = 2 }

  [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
  public class MMDeviceEnumeratorComObject {}

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("A95664D2-9614-4F35-A746-DE8DB63617E6")]
  public interface IMMDeviceEnumerator {
    [PreserveSig] int EnumAudioEndpoints(EDataFlow dataFlow, uint dwStateMask, IntPtr ppDevices);
    [PreserveSig] int GetDefaultAudioEndpoint(EDataFlow dataFlow, ERole role, out IMMDevice ppEndpoint);
    [PreserveSig] int GetDevice(string pwstrId, out IMMDevice ppDevice);
    [PreserveSig] int RegisterEndpointNotificationCallback(IntPtr pClient);
    [PreserveSig] int UnregisterEndpointNotificationCallback(IntPtr pClient);
  }

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("D666063F-1587-4E43-81F1-B948E807363F")]
  public interface IMMDevice {
    [PreserveSig] int Activate(ref Guid iid, uint dwClsCtx, IntPtr pActivationParams, [MarshalAs(UnmanagedType.IUnknown)] out object ppInterface);
    [PreserveSig] int OpenPropertyStore(uint stgmAccess, IntPtr ppProperties);
    [PreserveSig] int GetId([MarshalAs(UnmanagedType.LPWStr)] out string ppstrId);
    [PreserveSig] int GetState(out uint pdwState);
  }

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F")]
  public interface IAudioSessionManager2 {
    [PreserveSig] int GetAudioSessionControl(IntPtr AudioSessionGuid, uint StreamFlags, out IAudioSessionControl SessionControl);
    [PreserveSig] int GetSimpleAudioVolume(IntPtr AudioSessionGuid, uint StreamFlags, out ISimpleAudioVolume AudioVolume);
    [PreserveSig] int GetSessionEnumerator(out IAudioSessionEnumerator SessionEnum);
    [PreserveSig] int RegisterSessionNotification(IntPtr SessionNotification);
    [PreserveSig] int UnregisterSessionNotification(IntPtr SessionNotification);
    [PreserveSig] int RegisterDuckNotification(string sessionID, IntPtr duckNotification);
    [PreserveSig] int UnregisterDuckNotification(IntPtr duckNotification);
  }

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("E2F5BB11-0570-40CA-ACDD-3AA01277DEE8")]
  public interface IAudioSessionEnumerator {
    [PreserveSig] int GetCount(out int SessionCount);
    [PreserveSig] int GetSession(int SessionCount, out IAudioSessionControl Session);
  }

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("F4B1A599-7266-4319-A8CA-E70ACB11E8CD")]
  public interface IAudioSessionControl {
    [PreserveSig] int GetState(out int pRetVal);
    [PreserveSig] int GetDisplayName([MarshalAs(UnmanagedType.LPWStr)] out string pRetVal);
    [PreserveSig] int SetDisplayName(string Value, Guid EventContext);
    [PreserveSig] int GetIconPath([MarshalAs(UnmanagedType.LPWStr)] out string pRetVal);
    [PreserveSig] int SetIconPath(string Value, Guid EventContext);
    [PreserveSig] int GetGroupingParam(out Guid pRetVal);
    [PreserveSig] int SetGroupingParam(Guid Override, Guid EventContext);
    [PreserveSig] int RegisterAudioSessionNotification(IntPtr NewNotifications);
    [PreserveSig] int UnregisterAudioSessionNotification(IntPtr NewNotifications);
  }

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D")]
  public interface IAudioSessionControl2 {
    [PreserveSig] int GetState(out int pRetVal);
    [PreserveSig] int GetDisplayName([MarshalAs(UnmanagedType.LPWStr)] out string pRetVal);
    [PreserveSig] int SetDisplayName(string Value, Guid EventContext);
    [PreserveSig] int GetIconPath([MarshalAs(UnmanagedType.LPWStr)] out string pRetVal);
    [PreserveSig] int SetIconPath(string Value, Guid EventContext);
    [PreserveSig] int GetGroupingParam(out Guid pRetVal);
    [PreserveSig] int SetGroupingParam(Guid Override, Guid EventContext);
    [PreserveSig] int RegisterAudioSessionNotification(IntPtr NewNotifications);
    [PreserveSig] int UnregisterAudioSessionNotification(IntPtr NewNotifications);
    [PreserveSig] int GetSessionIdentifier([MarshalAs(UnmanagedType.LPWStr)] out string pRetVal);
    [PreserveSig] int GetSessionInstanceIdentifier([MarshalAs(UnmanagedType.LPWStr)] out string pRetVal);
    [PreserveSig] int GetProcessId(out uint pRetVal);
    [PreserveSig] int IsSystemSoundsSession();
    [PreserveSig] int SetDuckingPreference(bool optOut);
  }

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("87CE5498-68D6-44E5-9215-6DA47EF883D8")]
  public interface ISimpleAudioVolume {
    [PreserveSig] int SetMasterVolume(float fLevel, Guid EventContext);
    [PreserveSig] int GetMasterVolume(out float pfLevel);
    [PreserveSig] int SetMute(bool bMute, Guid EventContext);
    [PreserveSig] int GetMute(out bool pbMute);
  }

  public class Program {
    public static string Dump() {
      var sb = new StringBuilder();
      var enumerator = (IMMDeviceEnumerator)(new MMDeviceEnumeratorComObject());
      IMMDevice dev;
      int hr = enumerator.GetDefaultAudioEndpoint(EDataFlow.eRender, ERole.eMultimedia, out dev);
      if (hr != 0 || dev == null) return "GetDefaultAudioEndpoint failed hr=0x" + hr.ToString("X8");

      Guid iid = typeof(IAudioSessionManager2).GUID;
      object obj;
      hr = dev.Activate(ref iid, 23, IntPtr.Zero, out obj);
      if (hr != 0 || obj == null) return "Activate IAudioSessionManager2 failed hr=0x" + hr.ToString("X8");

      var mgr = (IAudioSessionManager2)obj;
      IAudioSessionEnumerator sessions;
      hr = mgr.GetSessionEnumerator(out sessions);
      if (hr != 0 || sessions == null) return "GetSessionEnumerator failed hr=0x" + hr.ToString("X8");

      int count;
      sessions.GetCount(out count);
      sb.AppendLine("SessionCount=" + count);
      for (int i = 0; i < count; i++) {
        IAudioSessionControl control;
        sessions.GetSession(i, out control);
        if (control == null) continue;
        int state;
        string display;
        control.GetState(out state);
        control.GetDisplayName(out display);
        var control2 = control as IAudioSessionControl2;
        var simple = control as ISimpleAudioVolume;
        uint pid = 0;
        string processName = "";
        if (control2 != null) {
          control2.GetProcessId(out pid);
          try {
            if (pid > 0) processName = Process.GetProcessById((int)pid).ProcessName;
          } catch {}
        }
        float volume = -1;
        bool muted = false;
        if (simple != null) {
          simple.GetMasterVolume(out volume);
          simple.GetMute(out muted);
        }
        sb.AppendLine(String.Format("State={0} Volume={1:0}% Muted={2} PID={3} Process={4} Display={5}", state, volume * 100, muted, pid, processName, display));
      }
      return sb.ToString();
    }
  }
}
"@

Add-Type -TypeDefinition $source -Language CSharp
[AudioSessionsInspect.Program]::Dump()
