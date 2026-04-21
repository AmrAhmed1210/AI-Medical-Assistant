using MedicalAssistant.Presentation.Hubs;
using MedicalAssistant.Services_Abstraction.Contracts;
using Microsoft.AspNetCore.SignalR;

namespace MedicalAssistant.Services.Services;

public class NotificationService : INotificationService
{
    private readonly IHubContext<NotificationHub> _hub;

    public NotificationService(IHubContext<NotificationHub> hub)
    {
        _hub = hub;
    }

    public async Task NotifyDoctorCreated(string doctorName, string specialty)
    {
        await SendGroupEventAsync("Patient", "DoctorCreated", new
        {
            message = $"New doctor {doctorName} ({specialty}) joined!",
            type = "new_doctor",
            timestamp = DateTime.UtcNow
        });
    }

    public async Task NotifyProfileUpdated(int doctorId, string doctorName)
    {
        await _hub.Clients.All.SendAsync("DoctorUpdated", new
        {
            doctorId,
            doctorName,
            message = "Doctor profile has been updated",
            type = "profile_updated",
            timestamp = DateTime.UtcNow
        });
    }

    public async Task NotifyAppointmentChanged(int appointmentId, string status, string patientEmail)
    {
        await SendGroupEventAsync(patientEmail, "AppointmentUpdated", new
        {
            appointmentId,
            status,
            message = $"Your appointment is now {status}",
            type = "appointment_update",
            timestamp = DateTime.UtcNow
        });

        await SendNotificationAsync(
            patientEmail,
            "appointment_update",
            "Appointment updated",
            $"Your appointment is now {status}",
            new { appointmentId, status });
    }

    public async Task NotifyNewUserRegistered(int userId, string name, string email, string role)
    {
        await SendGroupEventAsync("Admin", "NewUserRegistered", new
        {
            userId,
            name,
            email,
            role,
            message = $"New {role} registered: {name}",
            timestamp = DateTime.UtcNow
        });
    }

    public async Task NotifyScheduleReady(int doctorId, string doctorName)
    {
        await SendGroupEventAsync($"Schedule_Doctor_{doctorId}", "ScheduleReady", new
        {
            doctorId,
            doctorName,
            message = $"Dr. {doctorName}'s schedule is now available!",
            type = "schedule_ready",
            timestamp = DateTime.UtcNow
        });
    }

    public async Task NotifyScheduleUpdated(int doctorId, string doctorName, bool isMobileEnabled)
    {
        var payload = new
        {
            doctorId,
            doctorName,
            isMobileEnabled,
            message = $"Dr. {doctorName}'s schedule has been updated.",
            type = "schedule_updated",
            timestamp = DateTime.UtcNow
        };

        await SendGroupEventAsync($"Schedule_Doctor_{doctorId}", "ScheduleUpdated", payload);
        
        await SendNotificationAsync(
            "Patient",
            "schedule_updated",
            "Schedule updated",
            $"Dr. {doctorName}'s schedule has been updated.",
            new { doctorId, doctorName, isMobileEnabled });
    }

    public async Task NotifyAppointmentConfirmed(int appointmentId, string patientEmail, string doctorName, string scheduledAt)
    {
        await SendGroupEventAsync(patientEmail, "AppointmentUpdated", new
        {
            appointmentId,
            status = "Confirmed",
            message = $"Your appointment with Dr. {doctorName} is confirmed.",
            type = "appointment_confirmed",
            scheduledAt,
            timestamp = DateTime.UtcNow
        });

        await SendNotificationAsync(
            patientEmail,
            "appointment_confirmed",
            "Appointment confirmed",
            $"Your appointment with Dr. {doctorName} is confirmed.",
            new { appointmentId, doctorName, scheduledAt });
    }

    public async Task NotifyDoctorNewAppointment(int appointmentId, string doctorEmail, string patientName, string scheduledAt)
    {
        await SendGroupEventAsync(doctorEmail, "AppointmentUpdated", new
        {
            appointmentId,
            status = "Confirmed",
            message = $"New booking from {patientName}.",
            type = "new_booking",
            scheduledAt,
            timestamp = DateTime.UtcNow
        });

        await SendNotificationAsync(
            doctorEmail,
            "new_booking",
            "New appointment booked",
            $"Patient {patientName} booked an appointment.",
            new { appointmentId, patientName, scheduledAt });
    }

    public async Task NotifyAppointmentReminder(int appointmentId, string patientEmail, string doctorName, string scheduledAt)
    {
        await SendNotificationAsync(
            patientEmail,
            "appointment_reminder",
            "Appointment reminder",
            $"Reminder: you have an appointment with Dr. {doctorName} soon.",
            new { appointmentId, doctorName, scheduledAt });
    }

    public async Task NotifyMissedAppointment(int appointmentId, string patientEmail, string doctorName)
    {
        await SendNotificationAsync(
            patientEmail,
            "missed_appointment",
            "Missed appointment",
            $"You missed your appointment with Dr. {doctorName}.",
            new { appointmentId, doctorName });

        await SendNotificationAsync(
            patientEmail,
            "rebook_offer",
            "Free rebook available",
            $"You can rebook your missed appointment with Dr. {doctorName} at no extra charge.",
            new { appointmentId, doctorName });
    }

    public async Task NotifyFreeRebookOffer(int appointmentId, string patientEmail, string doctorName, string scheduledAt)
    {
        await SendNotificationAsync(
            patientEmail,
            "rebook_confirmed",
            "Free rebook confirmed",
            $"Your free rebook with Dr. {doctorName} is confirmed.",
            new { appointmentId, doctorName, scheduledAt });
    }

    public async Task NotifyDoctorAppointmentChanged(int appointmentId, string doctorEmail, string status, string patientName)
    {
        await SendGroupEventAsync(doctorEmail, "AppointmentUpdated", new
        {
            appointmentId,
            status,
            message = $"Appointment for {patientName} is now {status}.",
            type = "appointment_update",
            timestamp = DateTime.UtcNow
        });

        await SendNotificationAsync(
            doctorEmail,
            "appointment_update",
            "Appointment status updated",
            $"Appointment for {patientName} is now {status}.",
            new { appointmentId, status, patientName });
    }

    private async Task SendNotificationAsync(string group, string category, string title, string message, object? data = null)
    {
        await SendGroupEventAsync(group, "NotificationReceived", new
        {
            id = Guid.NewGuid().ToString("N"),
            category,
            title,
            message,
            createdAt = DateTime.UtcNow,
            data
        });
    }

    private async Task SendGroupEventAsync(string group, string eventName, object payload)
    {
        if (string.IsNullOrWhiteSpace(group))
        {
            return;
        }

        var groups = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            group
        };

        var lower = group.Trim().ToLowerInvariant();
        if (!string.IsNullOrWhiteSpace(lower))
        {
            groups.Add(lower);
        }

        foreach (var targetGroup in groups)
        {
            await _hub.Clients.Group(targetGroup).SendAsync(eventName, payload);
        }
    }
}
