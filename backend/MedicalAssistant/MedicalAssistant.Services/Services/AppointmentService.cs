using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using System.Globalization;

namespace MedicalAssistant.Services.Services
{
    public class AppointmentService : IAppointmentService
    {
        private const string ReminderMarker = "[REMINDER_SENT]";
        private const string FreeRebookMarker = "[FREE_REBOOK]";

        private readonly IUnitOfWork _unitOfWork;
        private readonly INotificationService _notificationService;

        public AppointmentService(IUnitOfWork unitOfWork, INotificationService notificationService)
        {
            _unitOfWork = unitOfWork;
            _notificationService = notificationService;
        }

        public async Task<AppointmentDto> CreateAppointmentAsync(CreateAppointmentDto dto)
        {
            var appointment = new Appointment
            {
                PatientId = dto.PatientId,
                DoctorId = dto.DoctorId,
                Date = dto.Date,
                Time = dto.Time,
                PaymentMethod = dto.PaymentMethod,
                Status = "Pending",
                Notes = dto.Notes,
                CreatedAt = DateTime.UtcNow
            };

            await _unitOfWork.Appointments.AddAsync(appointment);
            await _unitOfWork.SaveChangesAsync();

            var saved = await _unitOfWork.Appointments.GetByIdWithDoctorAsync(appointment.Id);
            if (saved == null)
            {
                return MapToDto(appointment);
            }

            if (!string.IsNullOrWhiteSpace(saved.Patient?.Email))
            {
                await _notificationService.NotifyAppointmentConfirmed(
                    saved.Id,
                    saved.Patient.Email,
                    saved.Doctor?.Name ?? "your doctor",
                    BuildScheduledAt(saved));
            }

            if (!string.IsNullOrWhiteSpace(saved.Doctor?.User?.Email))
            {
                await _notificationService.NotifyDoctorNewAppointment(
                    saved.Id,
                    saved.Doctor.User!.Email,
                    saved.Patient?.FullName ?? "a patient",
                    BuildScheduledAt(saved),
                    saved.DoctorId);
                await _notificationService.NotifyDoctorNewBooking(
                    saved.Doctor.User!.Email,
                    saved.Patient?.FullName ?? "a patient",
                    BuildScheduledAt(saved),
                    saved.DoctorId);
            }

            return MapToDto(saved);
        }

        public async Task<AppointmentDto?> GetAppointmentByIdAsync(int id)
        {
            var appointment = await _unitOfWork.Appointments.GetByIdWithDoctorAsync(id);
            if (appointment == null) return null;

            await ApplyAppointmentLifecycleAsync(new List<Appointment> { appointment });
            return MapToDto(appointment);
        }

        public async Task<IEnumerable<AppointmentDto>> GetAppointmentsByPatientIdAsync(int patientId)
        {
            var appointments = (await _unitOfWork.Appointments.GetByPatientIdWithDoctorAsync(patientId)).ToList();
            await ApplyAppointmentLifecycleAsync(appointments);
            return appointments.Select(MapToDto);
        }

        public async Task<IEnumerable<AppointmentDto>> GetAppointmentsByDoctorIdAsync(int doctorId)
        {
            var appointments = (await _unitOfWork.Appointments.GetByDoctorIdAsync(doctorId)).ToList();
            await ApplyAppointmentLifecycleAsync(appointments);
            return appointments.Select(MapToDto);
        }

        public async Task<AppointmentDto?> UpdateAppointmentAsync(UpdateAppointmentDto dto)
        {
            var appointment = await _unitOfWork.Appointments.GetByIdWithDoctorAsync(dto.Id);
            if (appointment == null) return null;

            appointment.Status = dto.Status;
            appointment.Notes = dto.Notes;

            _unitOfWork.Appointments.Update(appointment);
            await _unitOfWork.SaveChangesAsync();

            if (!string.IsNullOrWhiteSpace(appointment.Patient?.Email))
            {
                await _notificationService.NotifyAppointmentChanged(
                    appointment.Id,
                    appointment.Status,
                    appointment.Patient.Email);
            }

            if (!string.IsNullOrWhiteSpace(appointment.Doctor?.User?.Email))
            {
                await _notificationService.NotifyDoctorAppointmentChanged(
                    appointment.Id,
                    appointment.Doctor.User!.Email,
                    appointment.Status,
                    appointment.Patient?.FullName ?? "a patient",
                    appointment.DoctorId);
                await _notificationService.NotifyDoctorCancellation(
                    appointment.Doctor.User!.Email,
                    appointment.Patient?.FullName ?? "a patient",
                    BuildScheduledAt(appointment),
                    appointment.DoctorId);
            }

            return MapToDto(appointment);
        }

        public async Task<bool> DeleteAppointmentAsync(int id)
        {
            var appointment = await _unitOfWork.Appointments.GetByIdWithDoctorAsync(id);
            if (appointment == null) return false;

            appointment.Status = "Cancelled";
            _unitOfWork.Appointments.Update(appointment);
            await _unitOfWork.SaveChangesAsync();

            if (!string.IsNullOrWhiteSpace(appointment.Patient?.Email))
            {
                await _notificationService.NotifyAppointmentChanged(
                    appointment.Id,
                    appointment.Status,
                    appointment.Patient.Email);
            }

            if (!string.IsNullOrWhiteSpace(appointment.Doctor?.User?.Email))
            {
                await _notificationService.NotifyDoctorAppointmentChanged(
                    appointment.Id,
                    appointment.Doctor.User!.Email,
                    appointment.Status,
                    appointment.Patient?.FullName ?? "a patient",
                    appointment.DoctorId);

                if (string.Equals(appointment.Status, "Cancelled", StringComparison.OrdinalIgnoreCase))
                {
                    await _notificationService.NotifyDoctorCancellation(
                        appointment.Doctor.User!.Email,
                        appointment.Patient?.FullName ?? "a patient",
                        BuildScheduledAt(appointment),
                        appointment.DoctorId);
                }
            }

            return true;
        }

        public async Task<AppointmentDto?> RebookAppointmentAsync(int appointmentId, int patientId)
        {
            var original = await _unitOfWork.Appointments.GetByIdWithDoctorAsync(appointmentId);
            if (original == null || original.PatientId != patientId)
            {
                return null;
            }

            if (!string.Equals(original.Status, "Missed", StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            var nextSlot = await FindNextAvailableSlotAsync(original.DoctorId, DateTime.UtcNow.AddMinutes(30));
            if (!nextSlot.HasValue)
            {
                return null;
            }

            var rebooked = new Appointment
            {
                PatientId = original.PatientId,
                DoctorId = original.DoctorId,
                Date = nextSlot.Value.ToString("d MMM yyyy", CultureInfo.InvariantCulture),
                Time = nextSlot.Value.ToString("hh:mm tt", CultureInfo.InvariantCulture),
                PaymentMethod = original.PaymentMethod,
                Status = "Confirmed",
                Notes = MergeNotes(original.Notes, $"{FreeRebookMarker} Rebooked from appointment #{original.Id}"),
                CreatedAt = DateTime.UtcNow
            };

            await _unitOfWork.Appointments.AddAsync(rebooked);
            await _unitOfWork.SaveChangesAsync();

            var saved = await _unitOfWork.Appointments.GetByIdWithDoctorAsync(rebooked.Id);
            if (saved == null)
            {
                return MapToDto(rebooked);
            }

            if (!string.IsNullOrWhiteSpace(saved.Patient?.Email))
            {
                await _notificationService.NotifyFreeRebookOffer(
                    saved.Id,
                    saved.Patient.Email,
                    saved.Doctor?.Name ?? "your doctor",
                    BuildScheduledAt(saved));
            }

            return MapToDto(saved);
        }

        public async Task<PaginatedResultDto<AppointmentDto>> GetPaginatedAppointmentsAsync(int pageNumber, int pageSize)
        {
            var (items, totalCount) = await _unitOfWork.Appointments.GetPaginatedAsync(pageNumber, pageSize);
            var appointments = items.ToList();
            await ApplyAppointmentLifecycleAsync(appointments);

            return new PaginatedResultDto<AppointmentDto>
            {
                Items = appointments.Select(MapToDto),
                TotalCount = totalCount,
                PageNumber = pageNumber,
                PageSize = pageSize
            };
        }

        private async Task ApplyAppointmentLifecycleAsync(List<Appointment> appointments)
        {
            if (appointments.Count == 0) return;

            var now = DateTime.UtcNow;
            var changed = false;

            foreach (var appointment in appointments)
            {
                var scheduledAt = ParseAppointmentDateTime(appointment.Date, appointment.Time);
                if (!scheduledAt.HasValue) continue;

                if (string.Equals(appointment.Status, "Confirmed", StringComparison.OrdinalIgnoreCase))
                {
                    if (scheduledAt.Value <= now)
                    {
                        appointment.Status = "Missed";
                        changed = true;

                        if (!string.IsNullOrWhiteSpace(appointment.Patient?.Email))
                        {
                            await _notificationService.NotifyMissedAppointment(
                                appointment.Id,
                                appointment.Patient.Email,
                                appointment.Doctor?.Name ?? "your doctor");
                        }
                    }
                    else if (scheduledAt.Value <= now.AddHours(24) && !HasMarker(appointment.Notes, ReminderMarker))
                    {
                        appointment.Notes = MergeNotes(appointment.Notes, ReminderMarker);
                        changed = true;

                        if (!string.IsNullOrWhiteSpace(appointment.Patient?.Email))
                        {
                            await _notificationService.NotifyAppointmentReminder(
                                appointment.Id,
                                appointment.Patient.Email,
                                appointment.Doctor?.Name ?? "your doctor",
                                scheduledAt.Value.ToString("O"));
                        }
                    }
                }
            }

            if (changed)
            {
                await _unitOfWork.SaveChangesAsync();
            }
        }

        private async Task<DateTime?> FindNextAvailableSlotAsync(int doctorId, DateTime fromUtc)
        {
            var availability = (await _unitOfWork.Repository<DoctorAvailability>()
                .FindAsync(item => item.DoctorId == doctorId && item.IsAvailable))
                .OrderBy(item => item.DayOfWeek)
                .ThenBy(item => item.StartTime)
                .ToList();

            if (availability.Count == 0)
            {
                return null;
            }

            for (var offset = 0; offset < 30; offset++)
            {
                var date = fromUtc.Date.AddDays(offset);
                var dayAvailability = availability
                    .Where(item => item.DayOfWeek == (byte)date.DayOfWeek)
                    .OrderBy(item => item.StartTime)
                    .ToList();

                foreach (var slot in dayAvailability)
                {
                    var candidate = date.Add(slot.StartTime);
                    if (candidate > fromUtc)
                    {
                        return candidate;
                    }
                }
            }

            return null;
        }

        private static AppointmentDto MapToDto(Appointment appointment)
        {
            var scheduledAt = BuildScheduledAt(appointment);
            var isFreeRebook = HasMarker(appointment.Notes, FreeRebookMarker);

            return new AppointmentDto
            {
                Id = appointment.Id,
                PatientId = appointment.PatientId,
                DoctorId = appointment.DoctorId,
                DoctorName = appointment.Doctor?.Name ?? string.Empty,
                PatientName = appointment.Patient?.FullName ?? string.Empty,
                PatientPhotoUrl = appointment.Patient?.ImageUrl,
                Specialty = appointment.Doctor?.Specialty?.Name ?? string.Empty,
                Date = appointment.Date,
                Time = appointment.Time,
                ScheduledAt = scheduledAt,
                PaymentMethod = appointment.PaymentMethod,
                Status = appointment.Status,
                Notes = CleanNotes(appointment.Notes),
                IsFreeRebook = isFreeRebook,
                CanRebook = string.Equals(appointment.Status, "Missed", StringComparison.OrdinalIgnoreCase)
            };
        }

        private static DateTime? ParseAppointmentDateTime(string date, string time)
        {
            if (DateTime.TryParse($"{date} {time}", CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out var combined))
            {
                return combined.ToUniversalTime();
            }

            if (DateTime.TryParse(date, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out var dateOnly))
            {
                return dateOnly.ToUniversalTime();
            }

            return null;
        }

        private static string BuildScheduledAt(Appointment appointment)
        {
            return ParseAppointmentDateTime(appointment.Date, appointment.Time)?.ToString("O")
                ?? $"{appointment.Date} {appointment.Time}";
        }

        private static bool HasMarker(string? notes, string marker)
        {
            return !string.IsNullOrWhiteSpace(notes)
                && notes.Contains(marker, StringComparison.OrdinalIgnoreCase);
        }

        private static string? MergeNotes(string? notes, string addition)
        {
            if (string.IsNullOrWhiteSpace(notes)) return addition;
            if (notes.Contains(addition, StringComparison.OrdinalIgnoreCase)) return notes;
            return $"{notes} {addition}".Trim();
        }

        private static string? CleanNotes(string? notes)
        {
            if (string.IsNullOrWhiteSpace(notes)) return notes;

            return notes
                .Replace(ReminderMarker, string.Empty, StringComparison.OrdinalIgnoreCase)
                .Replace(FreeRebookMarker, string.Empty, StringComparison.OrdinalIgnoreCase)
                .Trim();
        }
    }
}
