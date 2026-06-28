$ErrorActionPreference = "Stop"

$source = @"
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace AudioInspect {
  public enum EDataFlow { eRender = 0, eCapture = 1, eAll = 2 }
  public enum ERole { eConsole = 0, eMultimedia = 1, eCommunications = 2 }
  [Flags] public enum DEVICE_STATE : uint { ACTIVE = 0x1, DISABLED = 0x2, NOTPRESENT = 0x4, UNPLUGGED = 0x8, MASK_ALL = 0xF }

  [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
  public class MMDeviceEnumeratorComObject {}

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("A95664D2-9614-4F35-A746-DE8DB63617E6")]
  public interface IMMDeviceEnumerator {
    [PreserveSig]
    int EnumAudioEndpoints(EDataFlow dataFlow, DEVICE_STATE dwStateMask, out IMMDeviceCollection ppDevices);
    [PreserveSig]
    int GetDefaultAudioEndpoint(EDataFlow dataFlow, ERole role, out IMMDevice ppEndpoint);
    [PreserveSig]
    int GetDevice(string pwstrId, out IMMDevice ppDevice);
    [PreserveSig]
    int RegisterEndpointNotificationCallback(IntPtr pClient);
    [PreserveSig]
    int UnregisterEndpointNotificationCallback(IntPtr pClient);
  }

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("0BD7A1BE-7A1A-44DB-8397-C0D3CCDBCC58")]
  public interface IMMDeviceCollection {
    [PreserveSig]
    int GetCount(out uint pcDevices);
    [PreserveSig]
    int Item(uint nDevice, out IMMDevice ppDevice);
  }

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("D666063F-1587-4E43-81F1-B948E807363F")]
  public interface IMMDevice {
    [PreserveSig]
    int Activate(ref Guid iid, uint dwClsCtx, IntPtr pActivationParams, [MarshalAs(UnmanagedType.IUnknown)] out object ppInterface);
    [PreserveSig]
    int OpenPropertyStore(uint stgmAccess, out IPropertyStore ppProperties);
    [PreserveSig]
    int GetId([MarshalAs(UnmanagedType.LPWStr)] out string ppstrId);
    [PreserveSig]
    int GetState(out DEVICE_STATE pdwState);
  }

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
  public interface IPropertyStore {
    [PreserveSig]
    int GetCount(out uint cProps);
    [PreserveSig]
    int GetAt(uint iProp, out PROPERTYKEY pkey);
    [PreserveSig]
    int GetValue(ref PROPERTYKEY key, out PROPVARIANT pv);
    [PreserveSig]
    int SetValue(ref PROPERTYKEY key, ref PROPVARIANT propvar);
    [PreserveSig]
    int Commit();
  }

  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("5CDF2C82-841E-4546-9722-0CF74078229A")]
  public interface IAudioEndpointVolume {
    [PreserveSig]
    int RegisterControlChangeNotify(IntPtr pNotify);
    [PreserveSig]
    int UnregisterControlChangeNotify(IntPtr pNotify);
    [PreserveSig]
    int GetChannelCount(out uint pnChannelCount);
    [PreserveSig]
    int SetMasterVolumeLevel(float fLevelDB, Guid pguidEventContext);
    [PreserveSig]
    int SetMasterVolumeLevelScalar(float fLevel, Guid pguidEventContext);
    [PreserveSig]
    int GetMasterVolumeLevel(out float pfLevelDB);
    [PreserveSig]
    int GetMasterVolumeLevelScalar(out float pfLevel);
    [PreserveSig]
    int SetChannelVolumeLevel(uint nChannel, float fLevelDB, Guid pguidEventContext);
    [PreserveSig]
    int SetChannelVolumeLevelScalar(uint nChannel, float fLevel, Guid pguidEventContext);
    [PreserveSig]
    int GetChannelVolumeLevel(uint nChannel, out float pfLevelDB);
    [PreserveSig]
    int GetChannelVolumeLevelScalar(uint nChannel, out float pfLevel);
    [PreserveSig]
    int SetMute(bool bMute, Guid pguidEventContext);
    [PreserveSig]
    int GetMute(out bool pbMute);
    [PreserveSig]
    int GetVolumeStepInfo(out uint pnStep, out uint pnStepCount);
    [PreserveSig]
    int VolumeStepUp(Guid pguidEventContext);
    [PreserveSig]
    int VolumeStepDown(Guid pguidEventContext);
    [PreserveSig]
    int QueryHardwareSupport(out uint pdwHardwareSupportMask);
    [PreserveSig]
    int GetVolumeRange(out float pflVolumeMindB, out float pflVolumeMaxdB, out float pflVolumeIncrementdB);
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct PROPERTYKEY {
    public Guid fmtid;
    public uint pid;
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct PROPVARIANT {
    public ushort vt;
    public ushort wReserved1;
    public ushort wReserved2;
    public ushort wReserved3;
    public IntPtr p;
    public int p2;
  }

  public class Program {
    static PROPERTYKEY PKEY_Device_FriendlyName = new PROPERTYKEY { fmtid = new Guid("A45C254E-DF1C-4EFD-8020-67D146A850E0"), pid = 14 };
    static PROPERTYKEY PKEY_Device_DeviceDesc = new PROPERTYKEY { fmtid = new Guid("A45C254E-DF1C-4EFD-8020-67D146A850E0"), pid = 2 };

    static string PropString(IPropertyStore store, PROPERTYKEY key) {
      PROPVARIANT pv;
      store.GetValue(ref key, out pv);
      if (pv.vt == 31 && pv.p != IntPtr.Zero) return Marshal.PtrToStringUni(pv.p);
      return "";
    }

    static string StateName(DEVICE_STATE state) {
      if (state == DEVICE_STATE.ACTIVE) return "ACTIVE";
      return state.ToString();
    }

    public static string Dump() {
      var sb = new StringBuilder();
      var enumerator = (IMMDeviceEnumerator)(new MMDeviceEnumeratorComObject());
      string defaultRenderConsole = IdOrEmpty(enumerator, EDataFlow.eRender, ERole.eConsole);
      string defaultRenderMultimedia = IdOrEmpty(enumerator, EDataFlow.eRender, ERole.eMultimedia);
      string defaultRenderCommunications = IdOrEmpty(enumerator, EDataFlow.eRender, ERole.eCommunications);
      string defaultCaptureConsole = IdOrEmpty(enumerator, EDataFlow.eCapture, ERole.eConsole);
      string defaultCaptureMultimedia = IdOrEmpty(enumerator, EDataFlow.eCapture, ERole.eMultimedia);
      string defaultCaptureCommunications = IdOrEmpty(enumerator, EDataFlow.eCapture, ERole.eCommunications);

      DumpFlow(sb, enumerator, EDataFlow.eRender, defaultRenderConsole, defaultRenderMultimedia, defaultRenderCommunications);
      DumpFlow(sb, enumerator, EDataFlow.eCapture, defaultCaptureConsole, defaultCaptureMultimedia, defaultCaptureCommunications);
      return sb.ToString();
    }

    static string IdOrEmpty(IMMDeviceEnumerator enumerator, EDataFlow flow, ERole role) {
      IMMDevice dev;
      int hr = enumerator.GetDefaultAudioEndpoint(flow, role, out dev);
      if (hr != 0 || dev == null) return "";
      string id;
      dev.GetId(out id);
      return id ?? "";
    }

    static void DumpFlow(StringBuilder sb, IMMDeviceEnumerator enumerator, EDataFlow flow, string defConsole, string defMultimedia, string defComms) {
      IMMDeviceCollection collection;
      enumerator.EnumAudioEndpoints(flow, DEVICE_STATE.MASK_ALL, out collection);
      uint count;
      collection.GetCount(out count);
      sb.AppendLine(flow == EDataFlow.eRender ? "RENDER" : "CAPTURE");
      for (uint i = 0; i < count; i++) {
        IMMDevice dev;
        collection.Item(i, out dev);
        string id;
        dev.GetId(out id);
        DEVICE_STATE state;
        dev.GetState(out state);
        IPropertyStore store;
        dev.OpenPropertyStore(0, out store);
        string name = PropString(store, PKEY_Device_FriendlyName);
        string desc = PropString(store, PKEY_Device_DeviceDesc);
        string roles = "";
        if (id == defConsole) roles += " ConsoleDefault";
        if (id == defMultimedia) roles += " MultimediaDefault";
        if (id == defComms) roles += " CommunicationsDefault";
        string volumeText = "";
        if (state == DEVICE_STATE.ACTIVE) {
          Guid iid = typeof(IAudioEndpointVolume).GUID;
          object obj;
          int hr = dev.Activate(ref iid, 23, IntPtr.Zero, out obj);
          if (hr == 0 && obj != null) {
            var vol = (IAudioEndpointVolume)obj;
            float scalar;
            bool muted;
            vol.GetMasterVolumeLevelScalar(out scalar);
            vol.GetMute(out muted);
            volumeText = String.Format(" Volume={0:0}% Muted={1}", scalar * 100, muted);
          }
        }
        sb.AppendLine(String.Format("State={0} Name={1} Desc={2}{3}{4}", StateName(state), name, desc, roles, volumeText));
        sb.AppendLine("  Id=" + id);
      }
    }
  }
}
"@

Add-Type -TypeDefinition $source -Language CSharp
[AudioInspect.Program]::Dump()
