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

    public async Task NotifyProfileUpdated(int doctorId, string doctorName, string? photoUrl = null)
    {
        var payload = new
        {
            doctorId,
            doctorName,
            photoUrl,
            message = "Doctor profile has been updated",
            type = "profile_updated",
            timestamp = DateTime.UtcNow
        };

        await SendGroupEventAsync($"Schedule_Doctor_{doctorId}", "DoctorUpdated", payload);
        await SendGroupEventAsync("Admin", "DoctorUpdated", payload);
        await SendNotificationAsync("Admin", "profile_updated", "Profile Updated", $"Dr. {doctorName} updated their profile.", payload);
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

        await SendNotificationAsync("Admin", "new_user", "New User Registered", $"{role} {name} has joined.", new { userId, name, email, role });
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
        
        // Notify targeted followers only
        await SendNotificationAsync(
            $"Schedule_Doctor_{doctorId}",
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

    public async Task NotifyDoctorNewAppointment(int appointmentId, string doctorEmail, string patientName, string scheduledAt, int? doctorId = null)
    {
        await SendDoctorGroupEventAsync(doctorId, doctorEmail, "AppointmentUpdated", new
        {
            appointmentId,
            status = "Confirmed",
            message = $"New booking from {patientName}.",
            type = "new_booking",
            scheduledAt,
            timestamp = DateTime.UtcNow
        });

        await SendDoctorNotificationAsync(
            doctorId,
            doctorEmail,
            "new_booking",
            "New appointment booked",
            $"Patient {patientName} booked an appointment.",
            new { appointmentId, patientName, scheduledAt });
    }

    public async Task NotifyDoctorNewBooking(string doctorEmail, string patientName, string appointmentDate, int? doctorId = null)
    {
        await SendDoctorGroupEventAsync(doctorId, doctorEmail, "NewBooking", new
        {
            patientName,
            date = appointmentDate,
            message = $"{patientName} booked an appointment on {appointmentDate}",
            type = "new_booking",
            timestamp = DateTime.UtcNow
        });
    }

    public async Task NotifyDoctorCancellation(string doctorEmail, string patientName, string appointmentDate, int? doctorId = null)
    {
        await SendDoctorGroupEventAsync(doctorId, doctorEmail, "BookingCancelled", new
        {
            patientName,
            date = appointmentDate,
            message = $"{patientName} cancelled their appointment",
            type = "cancellation",
            timestamp = DateTime.UtcNow
        });
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

    public async Task NotifyNewMessage(string patientEmail, string doctorName, string message, int? sessionId = null, int? doctorId = null)
    {
        var payload = new
        {
            doctorName,
            message,
            sessionId,
            type = "message",
            timestamp = DateTime.UtcNow
        };

        await SendGroupEventAsync(patientEmail, "NewMessage", payload);
        await SendNotificationAsync(
            patientEmail,
            "message",
            "New message",
            $"Dr. {doctorName}: {message}",
            new { doctorName, message, sessionId });
    }

    public async Task NotifyDoctorNewMessage(string doctorEmail, string patientName, string message, int? sessionId = null, int? doctorId = null, int? patientId = null, string? patientPhotoUrl = null)
    {
        var payload = new
        {
            patientName,
            message,
            sessionId,
            patientId,
            patientPhotoUrl,
            type = "message",
            timestamp = DateTime.UtcNow
        };

        await SendDoctorGroupEventAsync(doctorId, doctorEmail, "NewMessage", payload);
        await SendDoctorNotificationAsync(
            doctorId,
            doctorEmail,
            "message",
            "New message",
            $"Patient {patientName}: {message}",
            new { patientName, message, sessionId });
    }

    public async Task NotifyDoctorAppointmentChanged(int appointmentId, string doctorEmail, string status, string patientName, int? doctorId = null)
    {
        await SendDoctorGroupEventAsync(doctorId, doctorEmail, "AppointmentUpdated", new
        {
            appointmentId,
            status,
            message = $"Appointment for {patientName} is now {status}.",
            type = "appointment_update",
            timestamp = DateTime.UtcNow
        });

        await SendDoctorNotificationAsync(
            doctorId,
            doctorEmail,
            "appointment_update",
            "Appointment status updated",
            $"Appointment for {patientName} is now {status}.",
            new { appointmentId, status, patientName });
    }

    public async Task NotifyNewConsultation(string patientEmail, string doctorName, string title, string scheduledAt, int consultationId)
    {
        var payload = new
        {
            doctorName,
            title,
            scheduledAt,
            consultationId,
            message = $"Dr. {doctorName} scheduled a consultation: {title}",
            type = "new_consultation",
            timestamp = DateTime.UtcNow
        };

        await SendGroupEventAsync(patientEmail, "NewConsultation", payload);
        await SendNotificationAsync(
            patientEmail,
            "new_consultation",
            "New Consultation",
            $"Dr. {doctorName} scheduled a consultation: {title}",
            new { doctorName, title, scheduledAt, consultationId });
    }

    public async Task NotifyNewDoctorApplication(string applicantName, string applicantEmail)
    {
        var payload = new
        {
            applicantName,
            applicantEmail,
            message = $"New doctor application from {applicantName} ({applicantEmail})",
            type = "new_doctor_application",
            timestamp = DateTime.UtcNow
        };

        // Notify Admin group via both event name and NotificationReceived
        await SendGroupEventAsync("Admin", "NewDoctorApplication", payload);
        await SendNotificationAsync(
            "Admin",
            "new_doctor_application",
            "New Doctor Application",
            $"Dr. {applicantName} has applied to join the platform.",
            payload);
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

    private async Task SendDoctorNotificationAsync(int? doctorId, string doctorEmail, string category, string title, string message, object? data = null)
    {
        var payload = new
        {
            id = Guid.NewGuid().ToString("N"),
            category,
            title,
            message,
            createdAt = DateTime.UtcNow,
            data
        };

        await SendDoctorGroupEventAsync(doctorId, doctorEmail, "NotificationReceived", payload);
    }

    private async Task SendDoctorGroupEventAsync(int? doctorId, string doctorEmail, string eventName, object payload)
    {
        if (doctorId.HasValue && doctorId.Value > 0)
        {
            await SendGroupEventAsync($"Doctor_{doctorId.Value}", eventName, payload);
        }

        if (!string.IsNullOrWhiteSpace(doctorEmail))
        {
            await SendGroupEventAsync(doctorEmail, eventName, payload);
        }
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
