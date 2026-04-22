namespace MedicalAssistant.Services_Abstraction.Contracts;

public interface INotificationService
{
    Task NotifyDoctorCreated(string doctorName, string specialty);
    Task NotifyProfileUpdated(int doctorId, string doctorName);
    Task NotifyAppointmentChanged(int appointmentId, string status, string patientEmail);
    Task NotifyNewUserRegistered(int userId, string name, string email, string role);
    Task NotifyScheduleReady(int doctorId, string doctorName);
    Task NotifyScheduleUpdated(int doctorId, string doctorName, bool isMobileEnabled);
    Task NotifyAppointmentConfirmed(int appointmentId, string patientEmail, string doctorName, string scheduledAt);
    Task NotifyDoctorNewAppointment(int appointmentId, string doctorEmail, string patientName, string scheduledAt, int? doctorId = null);
    Task NotifyAppointmentReminder(int appointmentId, string patientEmail, string doctorName, string scheduledAt);
    Task NotifyMissedAppointment(int appointmentId, string patientEmail, string doctorName);
    Task NotifyFreeRebookOffer(int appointmentId, string patientEmail, string doctorName, string scheduledAt);
    Task NotifyDoctorAppointmentChanged(int appointmentId, string doctorEmail, string status, string patientName, int? doctorId = null);
}
